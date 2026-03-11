#pragma once
/**
 * FuturesPricer C++ implementation for energy commodity futures.
 *
 * Computes:
 *   - Mark-to-market P&L
 *   - Calendar spread values
 *   - Crack spread values (3:2:1)
 *   - Portfolio-level MTM
 */

#include "curve_engine.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace commodities {

struct ContractSpec {
    std::string name;
    double contract_size;
    double tick_size;
    double tick_value;
};

inline const std::unordered_map<std::string, ContractSpec>& contract_specs() {
    static const std::unordered_map<std::string, ContractSpec> specs = {
        {"CL", {"WTI Crude Oil", 1000.0, 0.01, 10.0}},
        {"HO", {"Heating Oil", 42000.0, 0.0001, 4.20}},
        {"RB", {"RBOB Gasoline", 42000.0, 0.0001, 4.20}},
        {"NG", {"Natural Gas", 10000.0, 0.001, 10.0}},
    };
    return specs;
}

struct FuturesPosition {
    std::string ticker;
    std::string product;
    int num_contracts;
    int direction;  // +1 long, -1 short
    double entry_price;
    double contract_size;
};

class FuturesPricer {
public:
    explicit FuturesPricer(const ForwardCurve& curve) : curve_(curve) {}

    /**
     * Mark-to-market P&L for a single position.
     */
    double mark_to_market(const FuturesPosition& pos, double tenor) const {
        double current = curve_.forward_price(tenor);
        return (current - pos.entry_price) * pos.num_contracts
               * pos.contract_size * pos.direction;
    }

    /**
     * Calendar spread value.
     */
    double calendar_spread(double t1, double t2, const std::string& product,
                           int num_spreads = 1) const {
        auto it = contract_specs().find(product);
        double cs = (it != contract_specs().end()) ? it->second.contract_size : 1000.0;
        double f1 = curve_.forward_price(t1);
        double f2 = curve_.forward_price(t2);
        return (f2 - f1) * num_spreads * cs;
    }

    /**
     * 3:2:1 crack spread value.
     */
    double crack_spread_321(double cl_price, double ho_price,
                            double rb_price, int num_cracks = 1) const {
        double product_rev = 2.0 * rb_price * 42.0 + 1.0 * ho_price * 42.0;
        double crude_cost = 3.0 * cl_price;
        double crack = product_rev - crude_cost;
        return crack * num_cracks * 1000.0;
    }

    /**
     * Batch MTM for multiple positions.
     */
    std::vector<double> portfolio_mtm(
        const std::vector<FuturesPosition>& positions,
        const std::vector<double>& tenors
    ) const {
        std::vector<double> results;
        results.reserve(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            double t = (i < tenors.size()) ? tenors[i] : 0.25;
            results.push_back(mark_to_market(positions[i], t));
        }
        return results;
    }

private:
    ForwardCurve curve_;
};

} // namespace commodities
