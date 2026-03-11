#pragma once
/**
 * ForwardCurve C++ implementation for energy commodity futures.
 *
 * High-performance forward curve operations:
 *   - Forward price interpolation (log-linear)
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

class ForwardCurve {
public:
    ForwardCurve(const std::vector<double>& times,
                 const std::vector<double>& prices)
        : times_(times), prices_(prices) {
        if (times.size() != prices.size()) {
            throw std::invalid_argument("times and prices must have same length");
        }
        if (times.empty()) {
            throw std::invalid_argument("Curve must have at least one point");
        }
    }

    /**
     * Interpolate forward price at time t using log-linear interpolation.
     */
    double forward_price(double t) const {
        if (times_.size() == 1) return prices_[0];

        // Clamp to curve range
        if (t <= times_.front()) return prices_.front();
        if (t >= times_.back()) return prices_.back();

        // Find bracketing interval
        auto it = std::lower_bound(times_.begin(), times_.end(), t);
        size_t i = std::distance(times_.begin(), it);
        if (i == 0) i = 1;

        double t0 = times_[i - 1], t1 = times_[i];
        double p0 = prices_[i - 1], p1 = prices_[i];

        if (p0 <= 0.0 || p1 <= 0.0) {
            // Fallback to linear if prices non-positive
            double w = (t - t0) / (t1 - t0);
            return p0 + w * (p1 - p0);
        }

        // Log-linear interpolation
        double w = (t - t0) / (t1 - t0);
        return std::exp(std::log(p0) + w * (std::log(p1) - std::log(p0)));
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
        return ForwardCurve(times_, shifted);
    }

    /**
     * Check if curve is in contango (front < back).
     */
    bool is_contango() const {
        if (prices_.size() < 2) return false;
        return prices_.front() < prices_.back();
    }

    // Accessors
    const std::vector<double>& times() const { return times_; }
    const std::vector<double>& prices() const { return prices_; }
    size_t size() const { return times_.size(); }

private:
    std::vector<double> times_;
    std::vector<double> prices_;
};

} // namespace commodities
