#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "llcuda/inference_engine.hpp"
#include "llcuda/types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_llcuda, m) {
    m.doc() = "Local LLaMA CUDA - Python bindings for CUDA-accelerated LLM inference";

    // Status class
    py::class_<llcuda::Status>(m, "Status")
        .def(py::init<>())
        .def_readonly("success", &llcuda::Status::success)
        .def_readonly("message", &llcuda::Status::message)
        .def("__bool__", [](const llcuda::Status& s) { return s.success; })
        .def("__repr__", [](const llcuda::Status& s) {
            return s.success ? "Status(OK)" : "Status(Error: " + s.message + ")";
        });

    // ModelConfig
    py::class_<llcuda::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &llcuda::ModelConfig::model_path)
        .def_readwrite("gpu_layers", &llcuda::ModelConfig::gpu_layers)
        .def_readwrite("ctx_size", &llcuda::ModelConfig::ctx_size)
        .def_readwrite("batch_size", &llcuda::ModelConfig::batch_size)
        .def_readwrite("threads", &llcuda::ModelConfig::threads)
        .def_readwrite("use_mlock", &llcuda::ModelConfig::use_mlock)
        .def_readwrite("use_mmap", &llcuda::ModelConfig::use_mmap)
        .def_readwrite("sha256_hash", &llcuda::ModelConfig::sha256_hash);

    // InferRequest
    py::class_<llcuda::InferRequest>(m, "InferRequest")
        .def(py::init<>())
        .def_readwrite("prompt", &llcuda::InferRequest::prompt)
        .def_readwrite("max_tokens", &llcuda::InferRequest::max_tokens)
        .def_readwrite("temperature", &llcuda::InferRequest::temperature)
        .def_readwrite("top_p", &llcuda::InferRequest::top_p)
        .def_readwrite("top_k", &llcuda::InferRequest::top_k)
        .def_readwrite("seed", &llcuda::InferRequest::seed)
        .def_readwrite("stream", &llcuda::InferRequest::stream)
        .def_readwrite("stop_sequences", &llcuda::InferRequest::stop_sequences);

    // InferResult
    py::class_<llcuda::InferResult>(m, "InferResult")
        .def(py::init<>())
        .def_readonly("success", &llcuda::InferResult::success)
        .def_readonly("text", &llcuda::InferResult::text)
        .def_readonly("tokens_generated", &llcuda::InferResult::tokens_generated)
        .def_readonly("latency_ms", &llcuda::InferResult::latency_ms)
        .def_readonly("tokens_per_sec", &llcuda::InferResult::tokens_per_sec)
        .def_readonly("error_message", &llcuda::InferResult::error_message)
        .def("__repr__", [](const llcuda::InferResult& r) {
            if (r.success) {
                return "InferResult(tokens=" + std::to_string(r.tokens_generated) + 
                       ", latency=" + std::to_string(r.latency_ms) + "ms, " +
                       "throughput=" + std::to_string(r.tokens_per_sec) + " tok/s)";
            } else {
                return "InferResult(Error: " + r.error_message + ")";
            }
        });

    // LatencyMetrics
    py::class_<llcuda::LatencyMetrics>(m, "LatencyMetrics")
        .def(py::init<>())
        .def_readonly("mean_ms", &llcuda::LatencyMetrics::mean_ms)
        .def_readonly("p50_ms", &llcuda::LatencyMetrics::p50_ms)
        .def_readonly("p95_ms", &llcuda::LatencyMetrics::p95_ms)
        .def_readonly("p99_ms", &llcuda::LatencyMetrics::p99_ms)
        .def_readonly("min_ms", &llcuda::LatencyMetrics::min_ms)
        .def_readonly("max_ms", &llcuda::LatencyMetrics::max_ms)
        .def_readonly("sample_count", &llcuda::LatencyMetrics::sample_count);

    // ThroughputMetrics
    py::class_<llcuda::ThroughputMetrics>(m, "ThroughputMetrics")
        .def(py::init<>())
        .def_readonly("total_tokens", &llcuda::ThroughputMetrics::total_tokens)
        .def_readonly("total_requests", &llcuda::ThroughputMetrics::total_requests)
        .def_readonly("tokens_per_sec", &llcuda::ThroughputMetrics::tokens_per_sec)
        .def_readonly("requests_per_sec", &llcuda::ThroughputMetrics::requests_per_sec);

    // GPUMetrics
    py::class_<llcuda::GPUMetrics>(m, "GPUMetrics")
        .def(py::init<>())
        .def_readonly("vram_used_mb", &llcuda::GPUMetrics::vram_used_mb)
        .def_readonly("vram_total_mb", &llcuda::GPUMetrics::vram_total_mb)
        .def_readonly("gpu_utilization", &llcuda::GPUMetrics::gpu_utilization)
        .def_readonly("temperature_c", &llcuda::GPUMetrics::temperature_c);

    // SystemMetrics
    py::class_<llcuda::SystemMetrics>(m, "SystemMetrics")
        .def(py::init<>())
        .def_readonly("latency", &llcuda::SystemMetrics::latency)
        .def_readonly("throughput", &llcuda::SystemMetrics::throughput)
        .def_readonly("gpu", &llcuda::SystemMetrics::gpu);

    // InferenceEngine
    py::class_<llcuda::InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load_model", &llcuda::InferenceEngine::load_model,
             py::arg("model_path"), py::arg("config") = llcuda::ModelConfig(),
             "Load a GGUF model for inference")
        .def("infer", &llcuda::InferenceEngine::infer,
             py::arg("request"),
             "Run inference on a single prompt")
        .def("infer_stream", 
             [](llcuda::InferenceEngine& self, const llcuda::InferRequest& req, py::function callback) {
                 return self.infer_stream(req, [callback](const std::string& chunk) {
                     py::gil_scoped_acquire acquire;
                     callback(chunk);
                 });
             },
             py::arg("request"), py::arg("callback"),
             "Run streaming inference with callback for each chunk")
        .def("infer_batch", &llcuda::InferenceEngine::infer_batch,
             py::arg("requests"),
             "Run batch inference on multiple prompts")
        .def("unload_model", &llcuda::InferenceEngine::unload_model,
             "Unload the current model")
        .def("is_model_loaded", &llcuda::InferenceEngine::is_model_loaded,
             "Check if a model is currently loaded")
        .def("get_metrics", &llcuda::InferenceEngine::get_metrics,
             "Get current performance metrics")
        .def("reset_metrics", &llcuda::InferenceEngine::reset_metrics,
             "Reset performance metrics counters");

    // Utility functions
    m.def("now_ms", &llcuda::now_ms, "Get current timestamp in milliseconds");

    // Version info
    m.attr("__version__") = "0.1.0";
}
