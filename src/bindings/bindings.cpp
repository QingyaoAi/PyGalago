// BSD License (http://www.galagosearch.org/license)
// pybind11 bindings for the galago C++ library — Phase 1 & Phase 2.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "galago/compression/vbyte.h"
#include "galago/btree/disk_btree_reader.h"
#include "galago/btree/disk_btree_iterator.h"
#include "galago/index/names_reader.h"
#include "galago/index/lengths_reader.h"
#include "galago/index/postings_reader.h"
#include "galago/index/disk_index.h"

namespace py = pybind11;
using namespace galago;

// ── B-tree iterator wrapper ───────────────────────────────────────────────────

struct PyBTreeIterator {
    DiskBTreeIterator it;

    bool        is_done()     const { return it.is_done(); }
    bool        next_key()          { return it.next_key(); }
    std::string key()         const { return it.key_string(); }
    py::bytes   key_bytes()   const {
        const auto& k = it.key();
        return py::bytes(reinterpret_cast<const char*>(k.data()), k.size());
    }
    py::bytes   value()       const {
        auto v = it.value_bytes();
        return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
    }
    int64_t value_start()  const { return it.value_start(); }
    int64_t value_end()    const { return it.value_end(); }
    int64_t value_length() const { return it.value_length(); }

    PyBTreeIterator& __iter__() { return *this; }
    py::object __next__() {
        if (it.is_done()) throw py::stop_iteration();
        auto k = key();
        auto v = value();
        it.next_key();
        return py::make_tuple(k, v);
    }
};

// ── PostingsIterator wrapper ──────────────────────────────────────────────────

struct PyPostingsIterator {
    PostingsIterator it;

    bool    is_done()     const { return it.is_done(); }
    int64_t doc_id()      const { return it.doc_id(); }
    int32_t count()       const { return it.count(); }
    void    next()              { it.next(); }
    void    skip_to(int64_t d)  { it.skip_to(d); }

    py::dict stats() const {
        const auto& s = it.stats();
        py::dict d;
        d["term"]             = s.term;
        d["document_count"]   = s.document_count;
        d["collection_count"] = s.collection_count;
        d["max_tf"]           = s.max_tf;
        return d;
    }

    PyPostingsIterator& __iter__() { return *this; }
    py::object __next__() {
        if (it.is_done()) throw py::stop_iteration();
        auto doc = it.doc_id();
        auto cnt = it.count();
        it.next();
        return py::make_tuple(doc, cnt);
    }
};

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_galago, m) {
    m.doc() = "PyGalago C++ extension — Galago index reader (Phase 1 + Phase 2).";

    // ── VByte ─────────────────────────────────────────────────────────────────
    m.def("vbyte_encode_u32", [](uint32_t v) {
        auto enc = vbyte_encode_u32(v);
        return py::bytes(reinterpret_cast<const char*>(enc.data()), enc.size());
    }, "VByte-encode a uint32 → bytes");

    m.def("vbyte_decode_u32", [](py::bytes data) {
        auto s = static_cast<std::string>(data);
        size_t off = 0;
        return vbyte_decode_u32(reinterpret_cast<const uint8_t*>(s.data()), off);
    }, "VByte-decode bytes → uint32");

    // ── BTreeIterator ─────────────────────────────────────────────────────────
    py::class_<PyBTreeIterator>(m, "BTreeIterator")
        .def_property_readonly("is_done",      &PyBTreeIterator::is_done)
        .def("next_key",    &PyBTreeIterator::next_key)
        .def_property_readonly("key",          &PyBTreeIterator::key)
        .def_property_readonly("key_bytes",    &PyBTreeIterator::key_bytes)
        .def_property_readonly("value",        &PyBTreeIterator::value)
        .def_property_readonly("value_start",  &PyBTreeIterator::value_start)
        .def_property_readonly("value_end",    &PyBTreeIterator::value_end)
        .def_property_readonly("value_length", &PyBTreeIterator::value_length)
        .def("__iter__", &PyBTreeIterator::__iter__, py::return_value_policy::reference)
        .def("__next__", &PyBTreeIterator::__next__);

    // ── BTreeReader ───────────────────────────────────────────────────────────
    py::class_<DiskBTreeReader>(m, "BTreeReader")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def_static("is_btree", &DiskBTreeReader::is_btree, py::arg("path"))
        .def_property_readonly("manifest_json", &DiskBTreeReader::manifest_json)
        .def("iterator", [](const DiskBTreeReader& r) -> py::object {
            auto it = r.get_iterator();
            if (!it) return py::none();
            return py::cast(PyBTreeIterator{std::move(*it)});
        })
        .def("get", [](const DiskBTreeReader& r, const std::string& key) -> py::object {
            auto it = r.get_iterator(key);
            if (!it) return py::none();
            return py::cast(PyBTreeIterator{std::move(*it)});
        }, py::arg("key"));

    // ── PostingsIterator ──────────────────────────────────────────────────────
    py::class_<PyPostingsIterator>(m, "PostingsIterator")
        .def_property_readonly("is_done", &PyPostingsIterator::is_done)
        .def_property_readonly("doc_id",  &PyPostingsIterator::doc_id)
        .def_property_readonly("count",   &PyPostingsIterator::count)
        .def("next",    &PyPostingsIterator::next,
             "Advance to the next posting.")
        .def("skip_to", &PyPostingsIterator::skip_to, py::arg("docid"),
             "Skip forward to the first posting with docid >= target.")
        .def_property_readonly("stats",   &PyPostingsIterator::stats,
             "Return dict with term, document_count, collection_count, max_tf.")
        .def("__iter__", &PyPostingsIterator::__iter__,
             py::return_value_policy::reference)
        .def("__next__", &PyPostingsIterator::__next__);

    // ── NamesReader ───────────────────────────────────────────────────────────
    py::class_<NamesReader>(m, "NamesReader")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Open a Galago `names` B-tree file.")
        .def("get_name", &NamesReader::get_name, py::arg("docid"),
             "Return the document name string for an internal docid.")
        .def_property_readonly("manifest_json", &NamesReader::manifest_json);

    // ── LengthStats ───────────────────────────────────────────────────────────
    py::class_<LengthStats>(m, "LengthStats")
        .def_readonly("field_name",           &LengthStats::field_name)
        .def_readonly("total_document_count", &LengthStats::total_document_count)
        .def_readonly("non_zero_doc_count",   &LengthStats::non_zero_doc_count)
        .def_readonly("collection_length",    &LengthStats::collection_length)
        .def_readonly("avg_length",           &LengthStats::avg_length)
        .def_readonly("max_length",           &LengthStats::max_length)
        .def_readonly("min_length",           &LengthStats::min_length)
        .def_readonly("first_document",       &LengthStats::first_document)
        .def_readonly("last_document",        &LengthStats::last_document)
        .def("__repr__", [](const LengthStats& s) {
            return "<LengthStats field=" + s.field_name
                 + " N=" + std::to_string(s.total_document_count)
                 + " avg=" + std::to_string(s.avg_length) + ">";
        });

    // ── LengthsReader ─────────────────────────────────────────────────────────
    py::class_<LengthsReader>(m, "LengthsReader")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Open a Galago `lengths` B-tree file.")
        .def("get_length", &LengthsReader::get_length, py::arg("docid"),
             "Return the document length (number of tokens) for an internal docid.")
        .def("get_stats",  &LengthsReader::get_stats,
             py::arg("field") = "document",
             "Return LengthStats for a field.")
        .def("total_documents", &LengthsReader::total_documents,
             py::arg("field") = "document")
        .def_property_readonly("manifest_json", &LengthsReader::manifest_json);

    // ── PostingsReader ────────────────────────────────────────────────────────
    py::class_<PostingsReader>(m, "PostingsReader")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Open a Galago postings B-tree file.")
        .def("get_postings", [](PostingsReader& r, const std::string& term) -> py::object {
            auto it = r.get_postings(term);
            if (!it) return py::none();
            return py::cast(PyPostingsIterator{std::move(*it)});
        }, py::arg("term"),
           "Return a PostingsIterator for term, or None if not in index.")
        .def("get_stats", [](PostingsReader& r, const std::string& term) -> py::object {
            auto s = r.get_stats(term);
            if (!s) return py::none();
            py::dict d;
            d["term"]             = s->term;
            d["document_count"]   = s->document_count;
            d["collection_count"] = s->collection_count;
            return d;
        }, py::arg("term"))
        .def_property_readonly("manifest_json", &PostingsReader::manifest_json);

    // ── DiskIndex ─────────────────────────────────────────────────────────────
    py::class_<DiskIndex>(m, "DiskIndex")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Open a Galago index directory.")
        .def("get_name",   &DiskIndex::get_name,   py::arg("docid"))
        .def("get_length", &DiskIndex::get_length, py::arg("docid"))
        .def("get_length_stats", &DiskIndex::get_length_stats,
             py::arg("field") = "document")
        .def("total_documents", &DiskIndex::total_documents)
        .def("has_names",    &DiskIndex::has_names)
        .def("has_lengths",  &DiskIndex::has_lengths)
        .def("has_postings", &DiskIndex::has_postings,
             py::arg("part") = "postings.krovetz")
        .def("get_postings", [](DiskIndex& idx, const std::string& term,
                                const std::string& part) -> py::object {
            auto it = idx.get_postings(term, part);
            if (!it) return py::none();
            return py::cast(PyPostingsIterator{std::move(*it)});
        }, py::arg("term"), py::arg("part") = "postings.krovetz",
           "Return a PostingsIterator for term in the given postings part.")
        .def_property_readonly("path", &DiskIndex::path);
}
