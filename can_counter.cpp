/*
 * count stacked cans with ONE top camera.
 * 
 * Inputs (measured): D = lens -> top lids of the tallest stacks, floor = lens -> ground,
 *                    can height (135), pitch = height added per extra stacked can (130.5)
 * 
 * Calibration (key 1): frame freezes; click 3 points around the rim of 1-3 TOP lids, Enter.
 *   Clicks only POINT at the lid; the rim is measured from edges.  K = D * r_top.
 * 
 * Detection (every frame) - only radii that can exist are searched:
 *   for each stack height n:  Z_n = floor - can - (n-1)*pitch,   r_n = K / Z_n
 *   1. circle search in a narrow band around r_n
 *   2. rim edge searched ONLY between 0.88 r_n and 1.12 r_n on 72 rays -> robust circle fit;
 *      a real rim fits within ~1 px, so fit error must be <= max_fit_px (default 2.5 px)
 *   3. the whole disc (20-80% of the radius) must be silver - rejects circles through other cans
 *   4. upgrade check: a real rim at the next-bigger size around the same centre means this was
 *      the inner ring of a taller stack's lid
 * Position from the image centre:  X = (x - cx) * Z,  Y = (y - cy) * Z  -> stack tracking.
 * Count: total = sum of stack heights; per layer (top first) and missing follow.
 * 
 * Keys: 1 calibrate  r run/pause  x forget stacks  v diagnostics (accepted AND rejected)
 *       +/- sensitivity  d show search circles (grey) / rejected (red)  s screenshot  q quit
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <map>
#include <tuple>
#include <optional>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <cctype>
#include <cstdio>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

// ================================================================ Constants & Lookup Tables
constexpr int VERSION = 11;
const std::string WIN = "Can counter";
const int FONT = cv::FONT_HERSHEY_SIMPLEX;

constexpr int NUM_RAYS = 72;
static double COS_A[NUM_RAYS];
static double SIN_A[NUM_RAYS];

static int g_s_max = 70;

static const std::map<int, cv::Scalar> N_COL = {
    {1, cv::Scalar(255, 0, 255)},
    {2, cv::Scalar(0, 255, 255)},
    {3, cv::Scalar(0, 255, 0)},
    {4, cv::Scalar(255, 160, 0)},
    {5, cv::Scalar(0, 140, 255)}
};

void init_trig() {
    static bool initialized = false;
    if (initialized) return;
    const double step = 2.0 * CV_PI / NUM_RAYS;
    for (int i = 0; i < NUM_RAYS; ++i) {
        double ang = i * step;
        COS_A[i] = std::cos(ang);
        SIN_A[i] = std::sin(ang);
    }
    initialized = true;
}

// ================================================================ Data Structures
struct Args {
    double cam_dist_mm = 0.0;
    bool has_cam_dist = false;
    double floor_dist_mm = 0.0;
    bool has_floor_dist = false;
    double can_height_mm = 135.0;
    double pitch_mm = 130.5;
    std::optional<int> layers = std::nullopt;
    int extra_layers = 1;
    double band = 0.08;
    double rim_tol = 0.06;
    int param2 = 25;
    int s_max = 70;
    double silver_frac = 0.75;
    double edge_frac = 0.4;
    int min_rays = 40;
    int min_rays_partial = 22;
    double max_fit_px = 2.5;
    double stack_tol = 1.0;
    int min_hits = 8;
    int empty_frames = 20;
    int vote = 15;
    double switch_frac = 0.6;
    std::optional<cv::Point2d> centre = std::nullopt;
    std::string calib = "can_counter_click.json";
    int cam = 0;
    std::string backend = "v4l2";
    int width = 640;
    int height = 480;
    std::optional<double> exposure = std::nullopt;
    double manual_exp_flag = 1.0;
    bool no_autofocus = false;
};

struct Lid {
    double x = 0.0;
    double y = 0.0;
    double r = 0.0;
    int rays = 0;
    double rms = 0.0;
    int n = 0;
    double X = 0.0;
    double Y = 0.0;
};

struct RimResult {
    double cx = 0.0;
    double cy = 0.0;
    double r = 0.0;
    int n_in = 0;
    double rms = 0.0;
};

struct Calib {
    int version = VERSION;
    int W = 0;
    int H = 0;
    double K = 0.0;
    double r_top = 0.0;
    cv::Point2d centre{0.0, 0.0};
    int n_full = 1;
    int n_max = 1;
};

struct StackInfo {
    double X = 0.0;
    double Y = 0.0;
    std::deque<int> hist;
    int state = 0;
    int noobs = 0;
    std::optional<cv::Vec3d> lid = std::nullopt; // (x, y, r)
};

struct Pending {
    Lid m;
    int hits = 1;
    int age = 0;
};

struct Rejected {
    double x = 0.0;
    double y = 0.0;
    double rn = 0.0;
    int n = 0;
    std::string why;
};

struct MouseState {
    std::string mode = "idle"; // "idle" or "calib"
    cv::Mat img;
    std::vector<cv::Point2d> clicks;
    std::vector<cv::Vec3d> circles; // (x, y, r)
};

struct PrepResult {
    cv::Mat hsv;
    cv::Mat gray;
    cv::Mat edge;
};

struct DetectResult {
    std::vector<Lid> lids;
    std::vector<cv::Vec3d> cands;
    cv::Mat hsv;
    cv::Mat edge;
    std::vector<Rejected> rej;
};

struct SummaryResult {
    std::vector<int> heights;
    int total = 0;
    int missing = 0;
    int extra = 0;
    std::vector<int> layers;
};

struct ReportState {
    int total = -1;
    int missing = -1;
    int extra = -1;
    std::vector<int> heights;
    bool operator==(const ReportState& other) const {
        return total == other.total && missing == other.missing &&
               extra == other.extra && heights == other.heights;
    }
    bool operator!=(const ReportState& other) const {
        return !(*this == other);
    }
};

// ================================================================ Geometry
inline double z_of(int n, const Args& a) {
    // Lens -> top lid of a stack holding n cans (mm).
    return a.floor_dist_mm - a.can_height_mm - (n - 1) * a.pitch_mm;
}

std::optional<cv::Vec3d> circle3(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& p3) {
    double x1 = p1.x, y1 = p1.y;
    double x2 = p2.x, y2 = p2.y;
    double x3 = p3.x, y3 = p3.y;
    double d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    if (std::abs(d) < 1e-6) {
        return std::nullopt;
    }
    double s1 = x1 * x1 + y1 * y1;
    double s2 = x2 * x2 + y2 * y2;
    double s3 = x3 * x3 + y3 * y3;
    double ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d;
    double uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d;
    return cv::Vec3d(ux, uy, std::hypot(x1 - ux, y1 - uy));
}

// ================================================================ Images Preprocessing
PrepResult prep(const cv::Mat& frame) {
    // HSV, gray for circle search, and an edge map (brightness + colour edges).
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    PrepResult res;
    cv::cvtColor(frame, res.hsv, cv::COLOR_BGR2HSV);

    cv::Mat gray_raw, gray_clahe;
    cv::cvtColor(frame, gray_raw, cv::COLOR_BGR2GRAY);
    clahe->apply(gray_raw, gray_clahe);
    cv::medianBlur(gray_clahe, res.gray, 5);

    std::vector<cv::Mat> hsv_planes;
    cv::split(res.hsv, hsv_planes);

    res.edge = cv::Mat::zeros(frame.size(), CV_32F);
    for (int ch : {1, 2}) { // saturation (silver<->red) + value (silver<->dark)
        cv::Mat c_blur, c_float;
        cv::GaussianBlur(hsv_planes[ch], c_blur, cv::Size(5, 5), 0);
        c_blur.convertTo(c_float, CV_32F);

        cv::Mat gx, gy, g;
        cv::Sobel(c_float, gx, CV_32F, 1, 0, 3);
        cv::Sobel(c_float, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, g);
        res.edge += g;
    }
    return res;
}

// ================================================================ Rim Measurement
double compute_median(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    size_t n = vals.size();
    size_t mid = n / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    if (n % 2 != 0) {
        return vals[mid];
    } else {
        auto max_left = std::max_element(vals.begin(), vals.begin() + mid);
        return (*max_left + vals[mid]) / 2.0;
    }
}

std::tuple<double, double, double> fit_circle(const std::vector<double>& px, const std::vector<double>& py) {
    int N = static_cast<int>(px.size());
    cv::Mat A(N, 3, CV_64F);
    cv::Mat b(N, 1, CV_64F);
    for (int i = 0; i < N; ++i) {
        A.at<double>(i, 0) = px[i];
        A.at<double>(i, 1) = py[i];
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = px[i] * px[i] + py[i] * py[i];
    }
    cv::Mat sol;
    cv::solve(A, b, sol, cv::DECOMP_SVD);
    double u = sol.at<double>(0, 0);
    double v = sol.at<double>(1, 0);
    double w = sol.at<double>(2, 0);
    double cx = u / 2.0;
    double cy = v / 2.0;
    double r = std::sqrt(std::max(w + cx * cx + cy * cy, 1e-6));
    return {cx, cy, r};
}

std::optional<RimResult> robust_circle(const std::vector<double>& px, const std::vector<double>& py) {
    size_t n_pts = px.size();
    if (n_pts < 10) return std::nullopt;
    std::vector<bool> keep(n_pts, true);

    for (int iter = 0; iter < 4; ++iter) {
        std::vector<double> k_px, k_py;
        k_px.reserve(n_pts);
        k_py.reserve(n_pts);
        for (size_t i = 0; i < n_pts; ++i) {
            if (keep[i]) {
                k_px.push_back(px[i]);
                k_py.push_back(py[i]);
            }
        }
        if (k_px.size() < 10) return std::nullopt;

        auto [cx, cy, r] = fit_circle(k_px, k_py);

        std::vector<double> res(n_pts);
        std::vector<double> res_keep;
        res_keep.reserve(k_px.size());
        for (size_t i = 0; i < n_pts; ++i) {
            res[i] = std::abs(std::hypot(px[i] - cx, py[i] - cy) - r);
            if (keep[i]) {
                res_keep.push_back(res[i]);
            }
        }
        double med = compute_median(res_keep);
        double thresh = std::max(1.5, 3.0 * (med + 1e-6));
        for (size_t i = 0; i < n_pts; ++i) {
            keep[i] = (res[i] <= thresh);
        }
    }

    std::vector<double> k_px, k_py;
    k_px.reserve(n_pts);
    k_py.reserve(n_pts);
    for (size_t i = 0; i < n_pts; ++i) {
        if (keep[i]) {
            k_px.push_back(px[i]);
            k_py.push_back(py[i]);
        }
    }
    if (k_px.size() < 10) return std::nullopt;

    auto [cx, cy, r] = fit_circle(k_px, k_py);
    double sum_sq = 0.0;
    for (size_t i = 0; i < k_px.size(); ++i) {
        double diff = std::hypot(k_px[i] - cx, k_py[i] - cy) - r;
        sum_sq += diff * diff;
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(k_px.size()));

    RimResult res_out;
    res_out.cx = cx;
    res_out.cy = cy;
    res_out.r = r;
    res_out.n_in = static_cast<int>(k_px.size());
    res_out.rms = rms;
    return res_out;
}

std::optional<RimResult> band_rim(const cv::Mat& edge, double x, double y, double r0, double lo, double hi, const Args& a) {
    // Strongest edge between lo*r0 and hi*r0 on each of 72 rays -> robust circle.
    int h = edge.rows;
    int w = edge.cols;
    double start_r = lo * r0;
    double end_r = hi * r0;
    std::vector<double> radii;
    for (double r = start_r; r < end_r - 1e-9; r += 0.5) {
        radii.push_back(r);
    }
    int nr = static_cast<int>(radii.size());
    if (nr < 4) return std::nullopt;

    std::vector<bool> ray_ok(NUM_RAYS, true);
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        for (int ri = 0; ri < nr; ++ri) {
            int xi = static_cast<int>(std::round(x + radii[ri] * COS_A[ray]));
            int yi = static_cast<int>(std::round(y + radii[ri] * SIN_A[ray]));
            if (xi < 0 || xi >= w || yi < 0 || yi >= h) {
                ray_ok[ray] = false;
                break;
            }
        }
    }

    int ok_count = 0;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        if (ray_ok[ray]) ok_count++;
    }
    if (ok_count < 15) return std::nullopt;

    std::vector<int> ok_rays;
    std::vector<int> best_k;
    std::vector<double> peak_val;
    ok_rays.reserve(NUM_RAYS);
    best_k.reserve(NUM_RAYS);
    peak_val.reserve(NUM_RAYS);

    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        if (!ray_ok[ray]) continue;
        ok_rays.push_back(ray);
        float max_val = -1.0f;
        int max_k = -1;
        for (int ri = 0; ri < nr; ++ri) {
            int xi = static_cast<int>(std::round(x + radii[ri] * COS_A[ray]));
            int yi = static_cast<int>(std::round(y + radii[ri] * SIN_A[ray]));
            float val = edge.at<float>(yi, xi);
            if (val > max_val) {
                max_val = val;
                max_k = ri;
            }
        }
        best_k.push_back(max_k);
        peak_val.push_back(static_cast<double>(max_val));
    }

    double med_peak = compute_median(peak_val);
    double thresh = a.edge_frac * med_peak;

    std::vector<double> strong_px;
    std::vector<double> strong_py;
    for (size_t idx = 0; idx < ok_rays.size(); ++idx) {
        int ray = ok_rays[idx];
        int k = best_k[idx];
        double p = peak_val[idx];
        if (p >= thresh && k > 0 && k < nr - 1) {
            double rb = radii[k];
            strong_px.push_back(x + rb * COS_A[ray]);
            strong_py.push_back(y + rb * SIN_A[ray]);
        }
    }

    if (strong_px.size() < 15) {
        return std::nullopt;
    }
    return robust_circle(strong_px, strong_py);
}

std::optional<RimResult> rim_two_pass(const cv::Mat& edge, double x, double y, double r0, const Args& a, double lo = 0.88, double hi = 1.12) {
    auto m = band_rim(edge, x, y, r0, lo, hi, a);
    if (!m) return std::nullopt;
    auto m2 = band_rim(edge, m->cx, m->cy, m->r, 0.93, 1.07, a); // refine around the fitted circle
    return m2 ? m2 : m;
}

double silver_cover(const cv::Mat& hsv, double x, double y, double r) {
    // Fraction of points at 20-80% of the radius that are silver (low saturation).
    int h = hsv.rows;
    int w = hsv.cols;
    const double ring_factors[5] = {0.2, 0.35, 0.5, 0.65, 0.8};
    int total_ok = 0;
    int silver_ok = 0;
    for (double factor : ring_factors) {
        double cur_r = factor * r;
        for (int i = 0; i < NUM_RAYS; ++i) {
            int px = static_cast<int>(std::round(x + cur_r * COS_A[i]));
            int py = static_cast<int>(std::round(y + cur_r * SIN_A[i]));
            if (px >= 0 && px < w && py >= 0 && py < h) {
                total_ok++;
                uchar sat = hsv.at<cv::Vec3b>(py, px)[1];
                if (sat <= g_s_max) {
                    silver_ok++;
                }
            }
        }
    }
    if (total_ok < 40) {
        return 0.0;
    }
    return static_cast<double>(silver_ok) / static_cast<double>(total_ok);
}

std::pair<std::optional<Lid>, std::string> check_lid(
    const cv::Mat& edge, const cv::Mat& hsv,
    double x, double y, double rn, const Args& a,
    int min_rays, double lo = 0.88, double hi = 1.12,
    std::optional<double> tol = std::nullopt)
{
    // Is there a lid of radius ~rn around (x,y)? Returns (lid, "") or (nullopt, reason).
    auto raw = rim_two_pass(edge, x, y, rn, a, lo, hi);
    if (!raw) {
        return {std::nullopt, "no rim edge in the expected band"};
    }
    double mx = raw->cx;
    double my = raw->cy;
    double mr = raw->r;
    int n_in = raw->n_in;
    double rms = raw->rms;

    char buf[256];
    if (n_in < min_rays) {
        std::snprintf(buf, sizeof(buf), "too few rim rays: %d of 72 agree (need %d), r=%.1fpx",
                      n_in, min_rays, mr);
        return {std::nullopt, std::string(buf)};
    }
    if (rms > a.max_fit_px) {
        std::snprintf(buf, sizeof(buf), "not a clean rim: fit error %.1fpx > %.1fpx (r=%.1fpx)",
                      rms, a.max_fit_px, mr);
        return {std::nullopt, std::string(buf)};
    }
    double eff_tol = tol.value_or(a.rim_tol);
    if (std::abs(mr / rn - 1.0) > eff_tol) {
        std::snprintf(buf, sizeof(buf), "rim %.1fpx, expected %.1fpx +-%.0f%%",
                      mr, rn, eff_tol * 100.0);
        return {std::nullopt, std::string(buf)};
    }
    double cover = silver_cover(hsv, mx, my, mr);
    if (cover < a.silver_frac) {
        std::snprintf(buf, sizeof(buf), "disc not silver: %.0f%% silver (need %.0f%%), r=%.1fpx",
                      cover * 100.0, a.silver_frac * 100.0, mr);
        return {std::nullopt, std::string(buf)};
    }
    Lid lid;
    lid.x = mx;
    lid.y = my;
    lid.r = mr;
    lid.rays = n_in;
    lid.rms = rms;
    return {lid, ""};
}

// ================================================================ Calibration
std::optional<Calib> finish_calib(const cv::Mat& img, const std::vector<cv::Vec3d>& circles, const Args& a) {
    // Clicks only point at the lids; the rim is measured from edges.
    int H = img.rows;
    int W = img.cols;
    PrepResult p = prep(img);
    std::vector<double> rs;
    std::cout << "\nCalibration (rim measured from edges; clicks only point at the lid):\n";
    for (size_t i = 0; i < circles.size(); ++i) {
        double x = circles[i][0];
        double y = circles[i][1];
        double r = circles[i][2];
        auto [lid, why] = check_lid(p.edge, p.hsv, x, y, r, a, a.min_rays, 0.92, 1.22, 0.25);
        if (!lid) {
            std::cout << "  lid " << (i + 1) << ": clicked r=" << std::fixed << std::setprecision(1) << r
                      << "px - " << why << " - lid skipped (click it again)\n";
            continue;
        }
        std::cout << "  lid " << (i + 1) << ": clicked r=" << std::fixed << std::setprecision(1) << r
                  << "px -> measured rim r=" << lid->r << "px (" << lid->rays
                  << " rays agree, fit error " << lid->rms << "px)\n";
        rs.push_back(lid->r);
    }
    if (rs.empty()) {
        std::cout << "  No rim measured - calibration NOT changed. Press 1 and click again.\n";
        return std::nullopt;
    }
    double r_top = compute_median(rs);
    if (rs.size() > 1) {
        auto [min_it, max_it] = std::minmax_element(rs.begin(), rs.end());
        if ((*max_it - *min_it) / r_top > 0.05) {
            std::cout << "  WARNING: measured top lids differ by more than 5% - were they all TOP lids of full stacks?\n";
        }
    }
    double K = a.cam_dist_mm * r_top;
    std::cout << "  top lid radius " << std::fixed << std::setprecision(1) << r_top
              << "px  ->  K = " << std::setprecision(0) << a.cam_dist_mm << " x "
              << std::setprecision(1) << r_top << " = " << std::setprecision(0) << K << "\n";

    Calib c;
    c.version = VERSION;
    c.W = W;
    c.H = H;
    c.K = K;
    c.r_top = r_top;
    return c;
}

void apply_geometry(Calib& calib, const Args& a) {
    if (a.centre.has_value()) {
        calib.centre = *a.centre;
    } else {
        calib.centre = cv::Point2d(calib.W / 2.0, calib.H / 2.0);
    }
    double n_raw = 1.0 + (a.floor_dist_mm - a.cam_dist_mm - a.can_height_mm) / a.pitch_mm;
    calib.n_full = a.layers.has_value() ? *a.layers : std::max(1, static_cast<int>(std::round(n_raw)));
    int n_max = calib.n_full + a.extra_layers;
    while (n_max > 1 && z_of(n_max, a) <= 50.0) {
        n_max--;
    }
    calib.n_max = n_max;

    std::cout << "\nLayers: 1 + (" << std::fixed << std::setprecision(0) << a.floor_dist_mm
              << " - " << a.cam_dist_mm << " - " << a.can_height_mm << ") / "
              << std::setprecision(1) << a.pitch_mm << " = " << std::setprecision(2) << n_raw
              << "  ->  N = " << calib.n_full << "\n";
    if (std::abs(n_raw - std::round(n_raw)) > 0.25 && !a.layers.has_value()) {
        std::cout << "WARNING: not close to a whole number - re-measure the distances\n";
    }
    std::cout << "K = " << std::fixed << std::setprecision(0) << calib.K
              << ". Lid radius searched for each stack height:\n";
    for (int n = 1; n <= n_max; ++n) {
        double zn = z_of(n, a);
        double rn = calib.K / zn;
        std::cout << "  " << n << " can(s): lid " << std::setw(5) << std::setprecision(0) << zn
                  << " mm from lens -> radius " << std::setw(6) << std::setprecision(1) << rn
                  << "px  (search " << std::setprecision(0) << (rn * (1.0 - a.band)) << "-"
                  << (rn * (1.0 + a.band)) << "px)\n";
    }
}

bool save_calib(const std::string& path, const Calib& c) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Could not open " << path << " for writing\n";
        return false;
    }
    f << "{\n";
    f << " \"version\": " << c.version << ",\n";
    f << " \"size\": [\n  " << c.W << ",\n  " << c.H << "\n ],\n";
    f << " \"K\": " << std::fixed << std::setprecision(4) << c.K << ",\n";
    f << " \"r_top\": " << std::fixed << std::setprecision(4) << c.r_top << ",\n";
    f << " \"centre\": [\n  " << c.centre.x << ",\n  " << c.centre.y << "\n ],\n";
    f << " \"n_full\": " << c.n_full << ",\n";
    f << " \"n_max\": " << c.n_max << "\n";
    f << "}\n";
    return true;
}

std::optional<Calib> load_calib(const std::string& path, int W, int H) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return std::nullopt;
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    auto find_num = [&](const std::string& key) -> std::optional<double> {
        size_t pos = s.find("\"" + key + "\"");
        if (pos == std::string::npos) return std::nullopt;
        pos = s.find(':', pos);
        if (pos == std::string::npos) return std::nullopt;
        pos++;
        while (pos < s.size() && (std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == '"')) pos++;
        size_t end = pos;
        while (end < s.size() && (std::isdigit(static_cast<unsigned char>(s[end])) || s[end] == '.' || s[end] == '-' || s[end] == '+')) end++;
        if (end > pos) {
            try {
                return std::stod(s.substr(pos, end - pos));
            } catch (...) {}
        }
        return std::nullopt;
    };

    auto find_vec2 = [&](const std::string& key) -> std::optional<std::pair<double, double>> {
        size_t pos = s.find("\"" + key + "\"");
        if (pos == std::string::npos) return std::nullopt;
        pos = s.find('[', pos);
        if (pos == std::string::npos) return std::nullopt;
        pos++;
        size_t end1 = s.find(',', pos);
        if (end1 == std::string::npos) return std::nullopt;
        size_t end2 = s.find(']', end1);
        if (end2 == std::string::npos) return std::nullopt;
        try {
            double v1 = std::stod(s.substr(pos, end1 - pos));
            double v2 = std::stod(s.substr(end1 + 1, end2 - end1 - 1));
            return std::make_pair(v1, v2);
        } catch (...) {}
        return std::nullopt;
    };

    auto ver = find_num("version");
    auto size = find_vec2("size");
    auto K = find_num("K");
    auto r_top = find_num("r_top");

    if (!ver || !size || !K || !r_top) {
        std::cout << "Could not read " << path << ": invalid or incomplete JSON\n";
        return std::nullopt;
    }

    if (static_cast<int>(*ver) != VERSION || static_cast<int>(size->first) != W || static_cast<int>(size->second) != H) {
        std::cout << path << " is from another version/resolution - press 1 to calibrate\n";
        return std::nullopt;
    }

    Calib c;
    c.version = static_cast<int>(*ver);
    c.W = static_cast<int>(size->first);
    c.H = static_cast<int>(size->second);
    c.K = *K;
    c.r_top = *r_top;

    auto centre = find_vec2("centre");
    if (centre) {
        c.centre = cv::Point2d(centre->first, centre->second);
    } else {
        c.centre = cv::Point2d(W / 2.0, H / 2.0);
    }
    auto nf = find_num("n_full");
    if (nf) c.n_full = static_cast<int>(*nf);
    auto nm = find_num("n_max");
    if (nm) c.n_max = static_cast<int>(*nm);

    std::cout << "Loaded " << path << ": top lid radius " << std::fixed << std::setprecision(1)
              << c.r_top << "px, K = " << std::setprecision(0) << c.K << "\n";
    return c;
}

// ================================================================ Detection
DetectResult detect_lids(const cv::Mat& frame, const Calib& calib, const Args& a, int p2) {
    // Returns accepted lids, all search circles, hsv, edge map, rejected candidates with reasons.
    int H = frame.rows;
    int W = frame.cols;
    PrepResult prep_res = prep(frame);

    DetectResult result;
    result.hsv = prep_res.hsv;
    result.edge = prep_res.edge;

    std::vector<Lid> lids;
    for (int n = 1; n <= calib.n_max; ++n) {
        double rn = calib.K / z_of(n, a);
        if (rn < 8.0 || rn > 0.45 * std::min(H, W)) {
            continue;
        }
        std::vector<cv::Vec3f> circles;
        int min_r = static_cast<int>(rn * (1.0 - a.band));
        int max_r = static_cast<int>(std::ceil(rn * (1.0 + a.band)));
        double min_dist = std::max(10.0, 1.6 * rn);

        cv::HoughCircles(prep_res.gray, circles, cv::HOUGH_GRADIENT, 1.2, min_dist, 120, p2, min_r, max_r);

        for (const auto& c : circles) {
            double x = c[0];
            double y = c[1];
            double r = c[2];
            result.cands.emplace_back(x, y, r);

            auto [lid_opt, why] = check_lid(prep_res.edge, prep_res.hsv, x, y, rn, a, a.min_rays);
            if (!lid_opt) {
                Rejected rej_item;
                rej_item.x = x;
                rej_item.y = y;
                rej_item.rn = rn;
                rej_item.n = n;
                rej_item.why = why;
                result.rej.push_back(rej_item);
                continue;
            }
            Lid lid = *lid_opt;
            lid.n = n;
            for (int m = n + 1; m <= calib.n_max; ++m) { // inner ring of a taller stack's lid?
                double rm = calib.K / z_of(m, a);
                auto [up_opt, _] = check_lid(prep_res.edge, prep_res.hsv, lid.x, lid.y, rm, a, a.min_rays, 0.9, 1.1);
                if (up_opt) {
                    up_opt->n = m;
                    lid = *up_opt;
                }
            }
            if (lid.x - lid.r < 0 || lid.y - lid.r < 0 ||
                lid.x + lid.r >= W || lid.y + lid.r >= H) {
                Rejected rej_item;
                rej_item.x = x;
                rej_item.y = y;
                rej_item.rn = rn;
                rej_item.n = n;
                rej_item.why = "cut by image edge";
                result.rej.push_back(rej_item);
                continue;
            }
            lids.push_back(lid);
        }
    }

    // A small circle CONCENTRIC with a bigger detected lid is that lid's inner ring
    std::vector<Lid> no_inner_lids;
    for (const auto& s : lids) {
        bool is_inner = false;
        for (const auto& b : lids) {
            if (b.r > 1.15 * s.r && std::hypot(s.x - b.x, s.y - b.y) < 0.25 * b.r) {
                is_inner = true;
                break;
            }
        }
        if (!is_inner) {
            no_inner_lids.push_back(s);
        }
    }
    lids = std::move(no_inner_lids);

    // Same lid found twice -> keep the best
    std::sort(lids.begin(), lids.end(), [](const Lid& a, const Lid& b) {
        if (a.rays != b.rays) return a.rays > b.rays;
        return a.rms < b.rms;
    });

    std::vector<Lid> kept;
    for (const auto& m : lids) {
        bool keep = true;
        for (const auto& k : kept) {
            if (std::hypot(m.x - k.x, m.y - k.y) <= 0.5 * std::max(m.r, k.r)) {
                keep = false;
                break;
            }
        }
        if (keep) {
            kept.push_back(m);
        }
    }

    // Filter rejected: keep only those not near kept lids
    std::vector<Rejected> filtered_rej;
    for (const auto& q : result.rej) {
        bool keep = true;
        for (const auto& k : kept) {
            if (std::hypot(q.x - k.x, q.y - k.y) <= 0.5 * k.r) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered_rej.push_back(q);
        }
    }

    result.lids = std::move(kept);
    result.rej = std::move(filtered_rej);
    return result;
}

// ================================================================ Stack Tracking
inline void push_hist(std::deque<int>& hist, int val, int maxlen) {
    hist.push_back(val);
    while (static_cast<int>(hist.size()) > maxlen) {
        hist.pop_front();
    }
}

StackInfo new_stack(const Lid& m, const Args& a) {
    StackInfo s;
    s.X = m.X;
    s.Y = m.Y;
    s.state = m.n;
    s.noobs = 0;
    s.lid = cv::Vec3d(m.x, m.y, m.r);
    push_hist(s.hist, m.n, a.vote);
    return s;
}

std::pair<int, std::optional<cv::Vec3d>> recheck(
    const StackInfo& s, const Calib& calib, const Args& a,
    const cv::Mat& hsv, const cv::Mat& edge)
{
    // No lid matched this stack: check for a rim at its expected spot, current height and lower.
    int H = edge.rows;
    int W = edge.cols;
    double cx = calib.centre.x;
    double cy = calib.centre.y;
    int start_n = s.state;
    for (int n = start_n; n >= 1; --n) {
        double Z = z_of(n, a);
        if (Z <= 50.0) continue;
        double rn = calib.K / Z;
        double px = cx + s.X / Z;
        double py = cy + s.Y / Z;
        if (px - rn < 0 || py - rn < 0 || px + rn >= W || py + rn >= H) {
            continue;
        }
        auto [lid_opt, _] = check_lid(edge, hsv, px, py, rn, a, a.min_rays_partial);
        if (lid_opt && std::hypot(lid_opt->x - px, lid_opt->y - py) < 0.4 * rn) {
            return {n, cv::Vec3d(lid_opt->x, lid_opt->y, lid_opt->r)};
        }
    }
    return {0, std::nullopt};
}

void update_stacks(
    std::vector<StackInfo>& stacks,
    std::vector<Pending>& pending,
    std::vector<Lid>& lids,
    const Calib& calib,
    const Args& a,
    const cv::Mat& hsv,
    const cv::Mat& edge,
    std::vector<std::string>* log)
{
    double K = calib.K;
    double cx = calib.centre.x;
    double cy = calib.centre.y;
    double tol = a.stack_tol * K;

    for (auto& m : lids) {
        double Z = K / m.r;
        m.X = (m.x - cx) * Z;
        m.Y = (m.y - cy) * Z;
    }

    std::map<int, std::vector<Lid>> assign;
    std::vector<Lid> free;
    for (const auto& m : lids) { // each lid -> nearest stack by real X,Y
        int best_idx = -1;
        double min_d = tol;
        for (size_t i = 0; i < stacks.size(); ++i) {
            double d = std::hypot(m.X - stacks[i].X, m.Y - stacks[i].Y);
            if (d < min_d) {
                min_d = d;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx >= 0) {
            assign[best_idx].push_back(m);
        } else {
            free.push_back(m);
        }
    }

    for (size_t i = 0; i < stacks.size(); ++i) {
        auto& s = stacks[i];
        std::string how;
        auto it = assign.find(static_cast<int>(i));
        if (it != assign.end() && !it->second.empty()) {
            const auto& cand = it->second;
            auto best_it = std::max_element(cand.begin(), cand.end(),
                                            [](const Lid& a, const Lid& b) { return a.n < b.n; });
            const Lid& m = *best_it; // visible lid = highest lid in the stack
            push_hist(s.hist, m.n, a.vote);
            s.noobs = 0;
            s.lid = cv::Vec3d(m.x, m.y, m.r);
            if (m.n == s.state) {
                s.X = 0.9 * s.X + 0.1 * m.X;
                s.Y = 0.9 * s.Y + 0.1 * m.Y;
            }
            char buf[256];
            std::snprintf(buf, sizeof(buf), "lid r=%.1fpx (%d rays, fit %.1fpx) -> %d cans",
                          m.r, m.rays, m.rms, m.n);
            how = buf;
        } else {
            auto [n, p] = recheck(s, calib, a, hsv, edge);
            if (n > 0) {
                push_hist(s.hist, n, a.vote);
                s.noobs = 0;
                s.lid = p;
                char buf[256];
                std::snprintf(buf, sizeof(buf), "rim found at expected spot -> %d cans", n);
                how = buf;
            } else {
                s.noobs++;
                s.lid = std::nullopt;
                if (s.noobs >= a.empty_frames) {
                    push_hist(s.hist, 0, a.vote);
                }
                char buf[256];
                std::snprintf(buf, sizeof(buf), "no lid for %d frames", s.noobs);
                how = buf;
            }
        }

        std::map<int, int> counts;
        for (int val : s.hist) {
            counts[val]++;
        }
        int top = s.state;
        int cnt = 0;
        for (const auto& [val, count] : counts) {
            if (count > cnt) {
                cnt = count;
                top = val;
            }
        }
        if (top != s.state && static_cast<int>(s.hist.size()) >= 5 &&
            static_cast<double>(cnt) >= a.switch_frac * static_cast<double>(s.hist.size())) {
            s.state = top;
        }
        if (log != nullptr) {
            char log_buf[512];
            std::snprintf(log_buf, sizeof(log_buf), "  stack %d: %s  ->  %d cans",
                          static_cast<int>(i), how.c_str(), s.state);
            log->push_back(log_buf);
        }
    }

    for (const auto& m : free) { // candidates for new stacks
        int best_idx = -1;
        double min_d = tol;
        for (size_t j = 0; j < pending.size(); ++j) {
            double d = std::hypot(m.X - pending[j].m.X, m.Y - pending[j].m.Y);
            if (d < min_d) {
                min_d = d;
                best_idx = static_cast<int>(j);
            }
        }
        if (best_idx >= 0) {
            pending[best_idx].hits++;
            pending[best_idx].m = m;
        } else {
            Pending p;
            p.m = m;
            p.hits = 1;
            p.age = 0;
            pending.push_back(p);
        }
    }

    std::vector<Pending> next_pending;
    for (auto& p : pending) {
        p.age++;
        if (p.hits >= a.min_hits) {
            stacks.push_back(new_stack(p.m, a));
        } else if (p.age <= 3 * a.min_hits) {
            next_pending.push_back(p);
        }
    }
    pending = std::move(next_pending);
}

SummaryResult summary(const std::vector<StackInfo>& stacks, int n_full) {
    SummaryResult res;
    for (const auto& s : stacks) {
        res.heights.push_back(s.state);
        res.total += s.state;
        res.missing += std::max(0, n_full - s.state);
        res.extra += std::max(0, s.state - n_full);
    }
    res.layers.resize(n_full, 0);
    for (int L = 1; L <= n_full; ++L) {
        int threshold = n_full - L + 1;
        int count = 0;
        for (int h : res.heights) {
            if (h >= threshold) count++;
        }
        res.layers[L - 1] = count;
    }
    return res;
}

// ================================================================ Camera & CLI
std::unique_ptr<cv::VideoCapture> open_camera(const Args& a) {
    auto cap = std::make_unique<cv::VideoCapture>();
    bool opened = false;
#if defined(__linux__)
    if (a.backend == "v4l2") {
        opened = cap->open(a.cam, cv::CAP_V4L2);
        if (!opened || !cap->isOpened()) {
            opened = false;
        }
    }
#endif
    if (!opened) {
        opened = cap->open(a.cam);
    }
    if (!opened || !cap->isOpened()) {
        return nullptr;
    }
    cap->set(cv::CAP_PROP_FRAME_WIDTH, a.width);
    cap->set(cv::CAP_PROP_FRAME_HEIGHT, a.height);
    if (a.no_autofocus) {
        cap->set(cv::CAP_PROP_AUTOFOCUS, 0);
    }
    if (a.exposure.has_value()) {
        cap->set(cv::CAP_PROP_AUTO_EXPOSURE, a.manual_exp_flag);
        cap->set(cv::CAP_PROP_EXPOSURE, *a.exposure);
    }
    return cap;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --cam-dist-mm <val> --floor-dist-mm <val> [options]\n\n"
              << "Options:\n"
              << "  --cam-dist-mm <float>       (Required) Distance from lens to top lids of tallest stacks (mm)\n"
              << "  --floor-dist-mm <float>     (Required) Distance from lens to ground (mm)\n"
              << "  --can-height-mm <float>     Can height (default: 135.0)\n"
              << "  --pitch-mm <float>          Height added per extra stacked can (default: 130.5)\n"
              << "  --layers <int>              Override N full layers\n"
              << "  --extra-layers <int>        Also search stacks taller than N (default: 1)\n"
              << "  --band <float>              Circle search band around each expected radius (default: 0.08)\n"
              << "  --rim-tol <float>           Measured rim tolerance (default: 0.06)\n"
              << "  --param2 <int>              Circle search sensitivity (default: 25)\n"
              << "  --s-max <int>               Max saturation of silver lid pixel (default: 70)\n"
              << "  --silver-frac <float>       Share of disc that must be silver (default: 0.75)\n"
              << "  --edge-frac <float>         Ray edge fraction threshold (default: 0.4)\n"
              << "  --min-rays <int>            Rim rays agreeing (default: 40)\n"
              << "  --min-rays-partial <int>    Rim rays agreeing for partial lid (default: 22)\n"
              << "  --max-fit-px <float>        Max rim circle fit error in px (default: 2.5)\n"
              << "  --stack-tol <float>         Stack match radius in lid radii (default: 1.0)\n"
              << "  --min-hits <int>            Hits before pending stack confirmed (default: 8)\n"
              << "  --empty-frames <int>        Frames without lid before state=0 (default: 20)\n"
              << "  --vote <int>                History deque length for voting (default: 15)\n"
              << "  --switch-frac <float>       Fraction needed to switch state (default: 0.6)\n"
              << "  --centre <CX> <CY>          Optical centre override\n"
              << "  --calib <file>              Calibration JSON path (default: can_counter_click.json)\n"
              << "  --cam <int>                 Camera device index (default: 0)\n"
              << "  --backend <v4l2|any>        Camera backend (default: v4l2)\n"
              << "  --width <int>               Capture width (default: 640)\n"
              << "  --height <int>              Capture height (default: 480)\n"
              << "  --exposure <float>          Manual exposure value\n"
              << "  --manual-exp-flag <float>   Manual exposure flag (default: 1)\n"
              << "  --no-autofocus              Disable camera autofocus\n"
              << "  -h, --help                  Print this help message\n";
}

std::optional<Args> parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--cam-dist-mm" && i + 1 < argc) {
            a.cam_dist_mm = std::stod(argv[++i]);
            a.has_cam_dist = true;
        } else if (arg == "--floor-dist-mm" && i + 1 < argc) {
            a.floor_dist_mm = std::stod(argv[++i]);
            a.has_floor_dist = true;
        } else if (arg == "--can-height-mm" && i + 1 < argc) {
            a.can_height_mm = std::stod(argv[++i]);
        } else if (arg == "--pitch-mm" && i + 1 < argc) {
            a.pitch_mm = std::stod(argv[++i]);
        } else if (arg == "--layers" && i + 1 < argc) {
            a.layers = std::stoi(argv[++i]);
        } else if (arg == "--extra-layers" && i + 1 < argc) {
            a.extra_layers = std::stoi(argv[++i]);
        } else if (arg == "--band" && i + 1 < argc) {
            a.band = std::stod(argv[++i]);
        } else if (arg == "--rim-tol" && i + 1 < argc) {
            a.rim_tol = std::stod(argv[++i]);
        } else if (arg == "--param2" && i + 1 < argc) {
            a.param2 = std::stoi(argv[++i]);
        } else if (arg == "--s-max" && i + 1 < argc) {
            a.s_max = std::stoi(argv[++i]);
        } else if (arg == "--silver-frac" && i + 1 < argc) {
            a.silver_frac = std::stod(argv[++i]);
        } else if (arg == "--edge-frac" && i + 1 < argc) {
            a.edge_frac = std::stod(argv[++i]);
        } else if (arg == "--min-rays" && i + 1 < argc) {
            a.min_rays = std::stoi(argv[++i]);
        } else if (arg == "--min-rays-partial" && i + 1 < argc) {
            a.min_rays_partial = std::stoi(argv[++i]);
        } else if (arg == "--max-fit-px" && i + 1 < argc) {
            a.max_fit_px = std::stod(argv[++i]);
        } else if (arg == "--stack-tol" && i + 1 < argc) {
            a.stack_tol = std::stod(argv[++i]);
        } else if (arg == "--min-hits" && i + 1 < argc) {
            a.min_hits = std::stoi(argv[++i]);
        } else if (arg == "--empty-frames" && i + 1 < argc) {
            a.empty_frames = std::stoi(argv[++i]);
        } else if (arg == "--vote" && i + 1 < argc) {
            a.vote = std::stoi(argv[++i]);
        } else if (arg == "--switch-frac" && i + 1 < argc) {
            a.switch_frac = std::stod(argv[++i]);
        } else if (arg == "--centre" && i + 2 < argc) {
            double cx = std::stod(argv[++i]);
            double cy = std::stod(argv[++i]);
            a.centre = cv::Point2d(cx, cy);
        } else if (arg == "--calib" && i + 1 < argc) {
            a.calib = argv[++i];
        } else if (arg == "--cam" && i + 1 < argc) {
            a.cam = std::stoi(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            a.backend = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            a.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            a.height = std::stoi(argv[++i]);
        } else if (arg == "--exposure" && i + 1 < argc) {
            a.exposure = std::stod(argv[++i]);
        } else if (arg == "--manual-exp-flag" && i + 1 < argc) {
            a.manual_exp_flag = std::stod(argv[++i]);
        } else if (arg == "--no-autofocus") {
            a.no_autofocus = true;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return std::nullopt;
        }
    }
    if (!a.has_cam_dist || !a.has_floor_dist) {
        std::cerr << "Error: --cam-dist-mm and --floor-dist-mm are required arguments.\n";
        print_usage(argv[0]);
        return std::nullopt;
    }
    return a;
}

// ================================================================ Mouse Callback
void on_mouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* st = static_cast<MouseState*>(userdata);
    if (!st) return;
    if (st->mode == "calib" && event == cv::EVENT_LBUTTONDOWN) {
        st->clicks.emplace_back(static_cast<double>(x), static_cast<double>(y));
        if (st->clicks.size() == 3) {
            auto c = circle3(st->clicks[0], st->clicks[1], st->clicks[2]);
            st->clicks.clear();
            if (c) {
                st->circles.push_back(*c);
                std::cout << "  lid " << st->circles.size() << ": centre ("
                          << std::fixed << std::setprecision(0) << (*c)[0] << "," << (*c)[1] << ") r="
                          << std::setprecision(1) << (*c)[2] << "px\n";
            } else {
                std::cout << "  the 3 points are in a line - click again\n";
            }
        }
    }
}

// ================================================================ Main
int main(int argc, char* argv[]) {
    init_trig();

    auto args_opt = parse_args(argc, argv);
    if (!args_opt.has_value()) {
        return 1;
    }
    Args a = *args_opt;
    g_s_max = a.s_max;

    auto cap = open_camera(a);
    if (!cap) {
        std::cerr << "Cannot open camera " << a.cam << "\n";
        return 1;
    }

    cv::Mat frame;
    bool ok = cap->read(frame);
    if (!ok || frame.empty()) {
        std::cerr << "Cannot read from camera\n";
        return 1;
    }

    int H = frame.rows;
    int W = frame.cols;

    std::cout << "Camera " << a.cam << ": " << W << "x" << H
              << "   (can_counter_click.py version " << VERSION
              << ", edge-band rim, fit <= " << std::fixed << std::setprecision(1) << a.max_fit_px
              << "px, disc >= " << std::setprecision(0) << (a.silver_frac * 100.0) << "% silver)\n";
    std::cout << "Keys: 1 calibrate  r run/pause  x forget stacks  v diagnostics  +/- sensitivity  "
              << "d search circles  s screenshot  q quit\n";

    std::optional<Calib> calib = load_calib(a.calib, W, H);
    if (calib.has_value()) {
        apply_geometry(*calib, a);
    }

    MouseState st;
    st.mode = "idle";

    try {
        cv::namedWindow(WIN, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(WIN, on_mouse, &st);
    } catch (const cv::Exception& e) {
        std::cerr << "GUI error (display unavailable?): " << e.what() << "\n";
        return 1;
    }

    std::vector<StackInfo> stacks;
    std::vector<Pending> pending;
    int p2 = a.param2;
    bool running = false;
    bool show_cand = false;
    bool diag = false;
    std::optional<ReportState> last_report = std::nullopt;

    auto t_prev = std::chrono::steady_clock::now();

    while (true) {
        ok = cap->read(frame);
        if (!ok || frame.empty()) {
            std::cout << "Frame grab failed\n";
            break;
        }

        cv::Mat out;
        if (st.mode == "calib") {
            out = st.img.clone();
            for (const auto& c : st.circles) {
                cv::circle(out, cv::Point(static_cast<int>(std::round(c[0])), static_cast<int>(std::round(c[1]))),
                           static_cast<int>(std::round(c[2])), cv::Scalar(0, 255, 0), 2);
            }
            for (const auto& pt : st.clicks) {
                cv::circle(out, cv::Point(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y))),
                           3, cv::Scalar(0, 0, 255), -1);
            }
            cv::putText(out, "Click 3 points around the rim of a TOP lid", cv::Point(10, 25), FONT, 0.55, cv::Scalar(0, 255, 255), 2);
            std::string calib_status = "lids: " + std::to_string(st.circles.size()) + "   Enter/space=done  c=clear  Esc=cancel";
            cv::putText(out, calib_status, cv::Point(10, 50), FONT, 0.55, cv::Scalar(0, 255, 255), 2);
        } else if (running && calib.has_value()) {
            auto det = detect_lids(frame, *calib, a, p2);
            std::vector<std::string> log_entries;
            std::vector<std::string>* log_ptr = diag ? &log_entries : nullptr;

            update_stacks(stacks, pending, det.lids, *calib, a, det.hsv, det.edge, log_ptr);
            out = frame.clone();

            if (show_cand) {
                for (const auto& c : det.cands) {
                    cv::circle(out, cv::Point(static_cast<int>(std::round(c[0])), static_cast<int>(std::round(c[1]))),
                               static_cast<int>(std::round(c[2])), cv::Scalar(120, 120, 120), 1);
                }
                for (const auto& q : det.rej) {
                    cv::circle(out, cv::Point(static_cast<int>(std::round(q.x)), static_cast<int>(std::round(q.y))),
                               static_cast<int>(std::round(q.rn)), cv::Scalar(0, 0, 255), 1);
                }
            }

            if (diag) {
                std::cout << "\n--- " << det.cands.size() << " search circles, " << det.lids.size()
                          << " verified lids (sensitivity " << p2 << ")\n";
                for (const auto& m : det.lids) {
                    std::cout << "  ACCEPTED (" << std::fixed << std::setprecision(0) << m.x << "," << m.y << ") r="
                              << std::setprecision(1) << m.r << "px rays=" << m.rays
                              << " fit=" << m.rms << "px -> " << m.n << " cans\n";
                }
                for (const auto& q : det.rej) {
                    std::cout << "  rejected (" << std::fixed << std::setprecision(0) << q.x << "," << q.y
                              << ") as " << q.n << "-can lid: " << q.why << "\n";
                }
                for (const auto& l : log_entries) {
                    std::cout << l << "\n";
                }
                diag = false;
            }

            double cx = calib->centre.x;
            double cy = calib->centre.y;
            for (const auto& s : stacks) {
                int n = s.state;
                double x = 0.0, y = 0.0, r = 0.0;
                if (s.lid.has_value()) {
                    x = (*s.lid)[0];
                    y = (*s.lid)[1];
                    r = (*s.lid)[2];
                } else {
                    double Z = z_of(std::max(n, 1), a);
                    x = cx + s.X / Z;
                    y = cy + s.Y / Z;
                    r = calib->K / Z;
                }
                if (n > 0) {
                    cv::Scalar col(255, 255, 255);
                    auto it = N_COL.find(n);
                    if (it != N_COL.end()) {
                        col = it->second;
                    }
                    cv::circle(out, cv::Point(static_cast<int>(std::round(x)), static_cast<int>(std::round(y))),
                               static_cast<int>(std::round(r)), col, 2);
                    cv::putText(out, std::to_string(n),
                                cv::Point(static_cast<int>(std::round(x)) - 7, static_cast<int>(std::round(y)) + 8),
                                FONT, 0.8, col, 2);
                } else {
                    cv::drawMarker(out, cv::Point(static_cast<int>(std::round(x)), static_cast<int>(std::round(y))),
                                   cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 30, 2);
                }
            }

            for (const auto& p : pending) {
                const auto& m = p.m;
                cv::circle(out, cv::Point(static_cast<int>(std::round(m.x)), static_cast<int>(std::round(m.y))),
                           static_cast<int>(std::round(m.r)), cv::Scalar(150, 150, 150), 1);
            }

            auto summ = summary(stacks, calib->n_full);
            int nf = calib->n_full;
            std::string head = "CANS " + std::to_string(summ.total) + "/" +
                               std::to_string(static_cast<int>(stacks.size()) * nf) +
                               "  MISSING " + std::to_string(summ.missing);
            if (summ.extra > 0) {
                head += "  EXTRA " + std::to_string(summ.extra);
            }
            cv::Scalar head_col = (summ.missing == 0 && summ.extra == 0) ?
                                  cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(out, head, cv::Point(10, 30), FONT, 0.75, head_col, 2);

            std::string lay;
            for (size_t L = 1; L <= summ.layers.size(); ++L) {
                if (L > 1) lay += "  ";
                lay += "L" + std::to_string(L);
                if (L == 1) lay += "(top)";
                lay += ":" + std::to_string(summ.layers[L - 1]) + "/" + std::to_string(stacks.size());
            }
            cv::putText(out, lay, cv::Point(10, 55), FONT, 0.5, cv::Scalar(255, 255, 255), 1);

            ReportState cur_report;
            cur_report.total = summ.total;
            cur_report.missing = summ.missing;
            cur_report.extra = summ.extra;
            cur_report.heights = summ.heights;

            if (!last_report.has_value() || *last_report != cur_report) {
                std::time_t t_now = std::time(nullptr);
                std::tm tm_now;
                localtime_r(&t_now, &tm_now);
                char time_buf[32];
                std::strftime(time_buf, sizeof(time_buf), "[%H:%M:%S]", &tm_now);

                std::string heights_str;
                for (size_t i = 0; i < summ.heights.size(); ++i) {
                    if (i > 0) heights_str += " ";
                    heights_str += std::to_string(summ.heights[i]);
                }

                std::cout << time_buf << " cans " << summ.total << "/" << (static_cast<int>(stacks.size()) * nf)
                          << ", missing " << summ.missing;
                if (summ.extra > 0) {
                    std::cout << ", extra " << summ.extra;
                }
                std::cout << " | " << lay << " | stack heights: " << heights_str << std::endl;
                last_report = cur_report;
            }
        } else {
            out = frame.clone();
            std::string msg = calib.has_value() ? "Idle - press r to run" : "Press 1 to calibrate";
            cv::putText(out, msg, cv::Point(10, 30), FONT, 0.7, cv::Scalar(255, 255, 255), 2);
        }

        auto t_now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t_now - t_prev).count();
        double fps = 1.0 / std::max(dt, 1e-6);
        t_prev = t_now;
        char fps_buf[64];
        std::snprintf(fps_buf, sizeof(fps_buf), "%.0f fps  sens %d", fps, p2);
        cv::putText(out, fps_buf, cv::Point(W - 150, H - 10), FONT, 0.45, cv::Scalar(255, 255, 255), 1);

        cv::imshow(WIN, out);

        int k = cv::waitKey(1) & 0xFF;
        if (st.mode == "calib") {
            if ((k == 13 || k == 10 || k == 32) && !st.circles.empty()) {
                auto new_calib = finish_calib(st.img, st.circles, a);
                if (new_calib.has_value()) {
                    calib = new_calib;
                    apply_geometry(*calib, a);
                    if (save_calib(a.calib, *calib)) {
                        std::cout << "Saved " << a.calib << ". Press r to run.\n";
                    }
                    stacks.clear();
                    pending.clear();
                }
                st.mode = "idle";
            } else if (k == 'c') {
                st.clicks.clear();
                st.circles.clear();
            } else if (k == 27) {
                st.mode = "idle";
                std::cout << "Calibration cancelled\n";
            } else if (k == 'q') {
                break;
            }
            continue;
        }

        if (k == 'q' || k == 27) {
            break;
        } else if (k == '1') {
            running = false;
            st.mode = "calib";
            st.img = frame.clone();
            st.clicks.clear();
            st.circles.clear();
            std::cout << "\nCalibration: click 3 points around the rim of 1-3 TOP lids, then press Enter\n";
        } else if (k == 'r' && calib.has_value()) {
            running = !running;
            last_report = std::nullopt;
            std::cout << (running ? "RUNNING" : "PAUSED") << std::endl;
        } else if (k == 'x') {
            stacks.clear();
            pending.clear();
            std::cout << "Stacks forgotten - relearning from the current view\n";
        } else if (k == 'v') {
            diag = running;
            if (!running) {
                std::cout << "Diagnostics work while running (press r first)\n";
            }
        } else if (k == '+' || k == '=') {
            p2 += 2;
            std::cout << "sensitivity (param2) = " << p2 << std::endl;
        } else if (k == '-') {
            p2 = std::max(10, p2 - 2);
            std::cout << "sensitivity (param2) = " << p2 << std::endl;
        } else if (k == 'd') {
            show_cand = !show_cand;
        } else if (k == 's') {
            auto ts = static_cast<long long>(std::time(nullptr));
            std::string filename = "cans_" + std::to_string(ts) + ".png";
            cv::imwrite(filename, out);
            std::cout << "Saved " << filename << std::endl;
        }
    }

    cap->release();
    cv::destroyAllWindows();
    return 0;
}
