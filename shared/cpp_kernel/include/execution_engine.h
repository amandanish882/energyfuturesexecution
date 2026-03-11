#pragma once
/**
 * Execution engine C++ implementation for energy commodity futures.
 *
 * Almgren-Chriss optimal execution with NYMEX-specific parameters.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_map>

namespace commodities {

struct EnergyFuturesSpec {
    std::string name;
    double contract_size;
    double tick_size;
    double avg_daily_volume;
    double daily_vol_pct;
};

inline const std::unordered_map<std::string, EnergyFuturesSpec>& energy_futures_specs() {
    static const std::unordered_map<std::string, EnergyFuturesSpec> specs = {
        {"CL", {"WTI Crude Oil", 1000.0, 0.01, 350000.0, 0.022}},
        {"HO", {"Heating Oil", 42000.0, 0.0001, 120000.0, 0.020}},
        {"RB", {"RBOB Gasoline", 42000.0, 0.0001, 100000.0, 0.024}},
        {"NG", {"Natural Gas", 10000.0, 0.001, 250000.0, 0.035}},
    };
    return specs;
}

class ExecutionEngine {
public:
    ExecutionEngine(double eta = 0.1, double lambda = 0.05,
                    double gamma = 0.5, double risk_aversion = 1e-6)
        : eta_(eta), lambda_(lambda), gamma_(gamma),
          risk_aversion_(risk_aversion) {}

    /**
     * Compute optimal Almgren-Chriss execution trajectory.
     * Returns cumulative fraction executed at each time step.
     */
    std::vector<double> optimal_trajectory(
        const std::string& product,
        int num_contracts,
        int n_slices = 10
    ) const {
        auto it = energy_futures_specs().find(product);
        double sigma = (it != energy_futures_specs().end())
                       ? it->second.daily_vol_pct : 0.022;

        double kappa = std::sqrt(risk_aversion_ * sigma * sigma / eta_);
        double kT = kappa * 1.0;

        std::vector<double> cum_executed(n_slices + 1);
        for (int i = 0; i <= n_slices; ++i) {
            double tau = static_cast<double>(i) / n_slices;
            if (std::abs(std::sinh(kT)) < 1e-10) {
                cum_executed[i] = tau;  // linear fallback
            } else {
                double remaining = std::sinh(kappa * (1.0 - tau)) / std::sinh(kT);
                cum_executed[i] = 1.0 - remaining;
            }
        }
        return cum_executed;
    }

    /**
     * Estimate temporary market impact.
     */
    double temporary_impact(
        const std::string& product,
        double trade_rate,
        double market_volume
    ) const {
        auto it = energy_futures_specs().find(product);
        double sigma = (it != energy_futures_specs().end())
                       ? it->second.daily_vol_pct : 0.022;

        if (market_volume <= 0) return 0.0;
        return eta_ * sigma * std::pow(trade_rate / market_volume, gamma_);
    }

    /**
     * Estimate permanent market impact.
     */
    double permanent_impact(
        const std::string& product,
        double total_volume,
        double market_volume
    ) const {
        auto it = energy_futures_specs().find(product);
        double sigma = (it != energy_futures_specs().end())
                       ? it->second.daily_vol_pct : 0.022;

        if (market_volume <= 0) return 0.0;
        return lambda_ * sigma * (total_volume / market_volume);
    }

private:
    double eta_;
    double lambda_;
    double gamma_;
    double risk_aversion_;
};

} // namespace commodities
