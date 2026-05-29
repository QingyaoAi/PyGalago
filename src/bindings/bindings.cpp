// BSD License (http://www.galagosearch.org/license)
// pybind11 bindings for the galago C++ library.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/optional.h>

#include "galago/compression/vbyte.h"
#include "galago/btree/disk_btree_reader.h"
#include "galago/btree/disk_btree_iterator.h"

namespace py = pybind11;
using namespace galago;

// ── Python-friendly iterator wrapper ─────────────────────────────────────────

struct PyBTreeIterator {
    DiskBTreeIterator it;

    bool is_done()  const { return it.is_done(); }
    bool next_key()       { return it.next_key(); }
    std::string key()     const { return it.key_string(); }
    py::bytes key_bytes() const {
        const auto& k = it.key();
        return py::bytes(reinterpret_cast<const char*>(k.data()), k.size());
    }
    py::bytes value() const {
        auto v = it.value_bytes();
        return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
    }
    int64_t value_start()  const { return it.value_start(); }
    int64_t value_end()    const { return it.value_end(); }
    int64_t value_length() const { return it.value_length(); }

    // Support Python's iterator protocol
    PyBTreeIterator& __iter__() { return *this; }
    py::object __next__() {
        if (it.is_done()) throw py::stop_iteration();
        auto k = key();
        auto v = value();
        it.next_key();
        return py::make_tuple(k, v);
    }
};

// ── Module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_galago, m) {
    m.doc() = "PyGalago C++ extension: Galago B-tree index reader and compression utilities.";

    // ── VByte ─────────────────────────────────────────────────────────────────
    m.def("vbyte_encode_u32", [](uint32_t v) {
        auto enc = vbyte_encode_u32(v);
        return py::bytes(reinterpret_cast<const char*>(enc.data()), enc.size());
    }, "VByte-encode a uint32 → bytes");

    m.def("vbyte_decode_u32", [](py::bytes data) {
        auto s = static_cast<std::string>(data);
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(s.data());
        size_t offset = 0;
        return vbyte_decode_u32(ptr, offset);
    }, "VByte-decode bytes → uint32");

    // ── BTreeIterator ─────────────────────────────────────────────────────────
    py::class_<PyBTreeIterator>(m, "BTreeIterator")
        .def_property_readonly("is_done",  &PyBTreeIterator::is_done)
        .def("next_key",  &PyBTreeIterator::next_key,
             "Advance to next key. Returns False when exhausted.")
        .def_property_readonly("key",       &PyBTreeIterator::key,
             "Current key as str (UTF-8 decoded).")
        .def_property_readonly("key_bytes", &PyBTreeIterator::key_bytes,
             "Current key as raw bytes.")
        .def_property_readonly("value",     &PyBTreeIterator::value,
             "Current value as raw bytes.")
        .def_property_readonly("value_start",  &PyBTreeIterator::value_start)
        .def_property_readonly("value_end",    &PyBTreeIterator::value_end)
        .def_property_readonly("value_length", &PyBTreeIterator::value_length)
        .def("__iter__", &PyBTreeIterator::__iter__,
             py::return_value_policy::reference)
        .def("__next__", &PyBTreeIterator::__next__);

    // ── BTreeReader ───────────────────────────────────────────────────────────
    py::class_<DiskBTreeReader>(m, "BTreeReader")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Open a Galago disk B-tree index file.")
        .def_static("is_btree", &DiskBTreeReader::is_btree, py::arg("path"),
             "Return True if path is a valid Galago B-tree file.")
        .def_property_readonly("manifest_json", &DiskBTreeReader::manifest_json,
             "The index manifest as a JSON string.")
        .def("iterator", [](const DiskBTreeReader& r) -> py::object {
            auto it = r.get_iterator();
            if (!it) return py::none();
            return py::cast(PyBTreeIterator{std::move(*it)});
        }, "Return an iterator over all key-value pairs, or None if empty.")
        .def("get", [](const DiskBTreeReader& r, const std::string& key) -> py::object {
            auto it = r.get_iterator(key);
            if (!it) return py::none();
            return py::cast(PyBTreeIterator{std::move(*it)});
        }, py::arg("key"),
           "Return an iterator positioned at key, or None if the key is not present.");
}
