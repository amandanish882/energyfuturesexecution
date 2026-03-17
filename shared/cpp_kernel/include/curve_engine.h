#pragma once
/**
 * ForwardCurve C++ implementation for energy commodity futures.
 *
 * High-performance forward curve operations:
 *   - Forward price interpolation (log-linear or monotone convex)
 *   - Convenience yield extraction
 *   - Roll yield computation
 *   - Curve shifting / bumping
 */

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace commodities {

enum class InterpMethod { LOG_LINEAR, MONOTONE_CONVEX };

class ForwardCurve {
public:
    ForwardCurve(const std::vector<double>& times,
                 const std::vector<double>& prices,
                 const std::string& method = "monotone_convex")
        : times_(times), prices_(prices) {
        if (times.size() != prices.size()) {
            throw std::invalid_argument("times and prices must have same length");
        }
        if (times.empty()) {
            throw std::invalid_argument("Curve must have at least one point");
        }
        if (method == "log_linear") {
            method_ = InterpMethod::LOG_LINEAR;
        } else {
            method_ = InterpMethod::MONOTONE_CONVEX;
            build_monotone_spline();
        }
    }

    /**
     * Interpolate forward price at time t.
     */
    double forward_price(double t) const {
        if (times_.size() == 1) return prices_[0];

        // Clamp to curve range
        if (t <= times_.front()) return prices_.front();
        if (t >= times_.back()) return prices_.back();

        if (method_ == InterpMethod::LOG_LINEAR) {
            return interp_log_linear(t);
        } else {
            return interp_monotone_convex(t);
        }
    }

    /**
     * Extract implied convenience yield.
     * y = r + u - (1/T) * ln(F(T) / S)
     */
    double convenience_yield(double t, double spot, double r = 0.045,
                             double storage = 0.03) const {
        if (t <= 0.0) return 0.0;
        double fwd = forward_price(t);
        if (spot <= 0.0 || fwd <= 0.0) return 0.0;
        return r + storage - (1.0 / t) * std::log(fwd / spot);
    }

    /**
     * Compute annualised roll yield between two tenors.
     */
    double roll_yield(double t1, double t2) const {
        double f1 = forward_price(t1);
        double f2 = forward_price(t2);
        if (std::abs(f1) < 1e-10) return 0.0;
        double days = (t2 - t1) * 365.0;
        if (days < 1.0) days = 1.0;
        return (f1 - f2) / f1 * (365.0 / days);
    }

    /**
     * Shift all prices by a parallel amount.
     */
    ForwardCurve shift(double amount) const {
        std::vector<double> shifted(prices_.size());
        for (size_t i = 0; i < prices_.size(); ++i) {
            shifted[i] = prices_[i] + amount;
        }
        std::string m = (method_ == InterpMethod::LOG_LINEAR) ? "log_linear" : "monotone_convex";
        return ForwardCurve(times_, shifted, m);
    }

    /**
     * Check if curve is in contango (front < back).
     */
    bool is_contango() const {
        if (prices_.size() < 2) return false;
        return prices_.front() < prices_.back();
    }

    std::string interpolation_method() const {
        return (method_ == InterpMethod::LOG_LINEAR) ? "log_linear" : "monotone_convex";
    }

    // Accessors
    const std::vector<double>& times() const { return times_; }
    const std::vector<double>& prices() const { return prices_; }
    size_t size() const { return times_.size(); }

private:
    std::vector<double> times_;
    std::vector<double> prices_;
    InterpMethod method_;

    // Fritsch-Carlson monotone cubic spline coefficients
    std::vector<double> mc_m_;  // slopes at each node

    /**
     * Log-linear interpolation (piecewise constant forward rate).
     */
    double interp_log_linear(double t) const {
        auto it = std::lower_bound(times_.begin(), times_.end(), t);
        size_t i = std::distance(times_.begin(), it);
        if (i == 0) i = 1;

        double t0 = times_[i - 1], t1 = times_[i];
        double p0 = prices_[i - 1], p1 = prices_[i];

        if (p0 <= 0.0 || p1 <= 0.0) {
            double w = (t - t0) / (t1 - t0);
            return p0 + w * (p1 - p0);
        }

        double w = (t - t0) / (t1 - t0);
        return std::exp(std::log(p0) + w * (std::log(p1) - std::log(p0)));
    }

    /**
     * Monotone convex interpolation via Fritsch-Carlson cubic Hermite.
     *
     * Uses the Fritsch-Carlson (1980) algorithm to compute node slopes
     * that guarantee monotonicity within each interval. Evaluation uses
     * standard cubic Hermite basis functions.
     */
    void build_monotone_spline() {
        size_t n = times_.size();
        if (n < 2) { mc_m_.assign(n, 0.0); return; }

        // Step 1: compute secants
        std::vector<double> delta(n - 1);
        std::vector<double> h(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            h[i] = times_[i + 1] - times_[i];
            if (std::abs(h[i]) < 1e-15) h[i] = 1e-15;
            delta[i] = (prices_[i + 1] - prices_[i]) / h[i];
        }

        // Step 2: initial slopes (three-point formula at interior, one-sided at ends)
        mc_m_.resize(n);
        mc_m_[0] = delta[0];
        mc_m_[n - 1] = delta[n - 2];
        for (size_t i = 1; i < n - 1; ++i) {
            if (delta[i - 1] * delta[i] <= 0.0) {
                mc_m_[i] = 0.0;
            } else {
                mc_m_[i] = (delta[i - 1] + delta[i]) / 2.0;
            }
        }

        // Step 3: Fritsch-Carlson monotonicity correction
        for (size_t i = 0; i < n - 1; ++i) {
            if (std::abs(delta[i]) < 1e-30) {
                mc_m_[i] = 0.0;
                mc_m_[i + 1] = 0.0;
                continue;
            }
            double alpha = mc_m_[i] / delta[i];
            double beta = mc_m_[i + 1] / delta[i];

            // Fritsch-Carlson condition: alpha^2 + beta^2 <= 9
            double phi = alpha * alpha + beta * beta;
            if (phi > 9.0) {
                double tau = 3.0 / std::sqrt(phi);
                mc_m_[i] = tau * alpha * delta[i];
                mc_m_[i + 1] = tau * beta * delta[i];
            }
        }
    }

    /**
     * Evaluate the monotone cubic Hermite interpolant at t.
     */
    double interp_monotone_convex(double t) const {
        // Find bracketing interval
        auto it = std::lower_bound(times_.begin(), times_.end(), t);
        size_t i = std::distance(times_.begin(), it);
        if (i == 0) i = 1;

        double t0 = times_[i - 1], t1 = times_[i];
        double p0 = prices_[i - 1], p1 = prices_[i];
        double m0 = mc_m_[i - 1], m1 = mc_m_[i];

        double h = t1 - t0;
        if (std::abs(h) < 1e-15) return p0;

        double s = (t - t0) / h;
        double s2 = s * s;
        double s3 = s2 * s;

        // Cubic Hermite basis functions
        double h00 = 2 * s3 - 3 * s2 + 1;
        double h10 = s3 - 2 * s2 + s;
        double h01 = -2 * s3 + 3 * s2;
        double h11 = s3 - s2;

        return h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1;
    }
};

} // namespace commodities
