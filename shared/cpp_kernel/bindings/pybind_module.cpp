/**
 * pybind11 bindings for commodities C++ kernel.
 *
 * Exposes ForwardCurve, FuturesPricer, and ExecutionEngine to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../include/curve_engine.h"
#include "../include/futures_pricer.h"
#include "../include/execution_engine.h"

namespace py = pybind11;
using namespace commodities;

PYBIND11_MODULE(commodities_cpp, m) {
    m.doc() = "C++ kernel for energy commodity trading platform";

    // ForwardCurve
    py::class_<ForwardCurve>(m, "ForwardCurve")
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                       const std::string&>(),
             py::arg("times"), py::arg("prices"),
             py::arg("method") = "monotone_convex")
        .def("forward_price", &ForwardCurve::forward_price, py::arg("t"))
        .def("convenience_yield", &ForwardCurve::convenience_yield,
             py::arg("t"), py::arg("spot"),
             py::arg("r") = 0.045, py::arg("storage") = 0.03)
        .def("roll_yield", &ForwardCurve::roll_yield,
             py::arg("t1"), py::arg("t2"))
        .def("shift", &ForwardCurve::shift, py::arg("amount"))
        .def("is_contango", &ForwardCurve::is_contango)
        .def("interpolation_method", &ForwardCurve::interpolation_method)
        .def("times", &ForwardCurve::times)
        .def("prices", &ForwardCurve::prices)
        .def("size", &ForwardCurve::size);

    // FuturesPosition
    py::class_<FuturesPosition>(m, "FuturesPosition")
        .def(py::init<>())
        .def_readwrite("ticker", &FuturesPosition::ticker)
        .def_readwrite("product", &FuturesPosition::product)
        .def_readwrite("num_contracts", &FuturesPosition::num_contracts)
        .def_readwrite("direction", &FuturesPosition::direction)
        .def_readwrite("entry_price", &FuturesPosition::entry_price)
        .def_readwrite("contract_size", &FuturesPosition::contract_size);

    // FuturesPricer
    py::class_<FuturesPricer>(m, "FuturesPricer")
        .def(py::init<const ForwardCurve&>(), py::arg("curve"))
        .def("mark_to_market", &FuturesPricer::mark_to_market,
             py::arg("position"), py::arg("tenor"))
        .def("calendar_spread", &FuturesPricer::calendar_spread,
             py::arg("t1"), py::arg("t2"), py::arg("product"),
             py::arg("num_spreads") = 1)
        .def("crack_spread_321", &FuturesPricer::crack_spread_321,
             py::arg("cl_price"), py::arg("ho_price"), py::arg("rb_price"),
             py::arg("num_cracks") = 1)
        .def("portfolio_mtm", &FuturesPricer::portfolio_mtm,
             py::arg("positions"), py::arg("tenors"));

    // ExecutionEngine
    py::class_<ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<double, double, double, double>(),
             py::arg("eta") = 0.1, py::arg("lambda_") = 0.05,
             py::arg("gamma") = 0.5, py::arg("risk_aversion") = 1e-6)
        .def("optimal_trajectory", &ExecutionEngine::optimal_trajectory,
             py::arg("product"), py::arg("num_contracts"),
             py::arg("n_slices") = 10)
        .def("temporary_impact", &ExecutionEngine::temporary_impact,
             py::arg("product"), py::arg("trade_rate"), py::arg("market_volume"))
        .def("permanent_impact", &ExecutionEngine::permanent_impact,
             py::arg("product"), py::arg("total_volume"), py::arg("market_volume"));
}
