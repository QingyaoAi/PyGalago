// BSD License (http://www.galagosearch.org/license)
#include "galago/index/disk_index.h"
#include "galago/btree/disk_btree_reader.h"

#include <map>
#include <stdexcept>
#include <sys/stat.h>

namespace galago {

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool file_exists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool dir_exists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// ── Construction ──────────────────────────────────────────────────────────────

DiskIndex::DiskIndex(const std::string& index_path) : path_(index_path) {
    if (!dir_exists(index_path)) {
        throw std::runtime_error("DiskIndex: not a directory: " + index_path);
    }

    auto part = [&](const std::string& name) {
        return index_path + "/" + name;
    };

    if (file_exists(part("names")) && DiskBTreeReader::is_btree(part("names"))) {
        names_ = std::make_unique<NamesReader>(part("names"));
    }

    if (file_exists(part("lengths")) && DiskBTreeReader::is_btree(part("lengths"))) {
        lengths_ = std::make_unique<LengthsReader>(part("lengths"));
    }

    for (const auto& pname : {"postings", "postings.krovetz"}) {
        std::string p = part(pname);
        if (file_exists(p) && DiskBTreeReader::is_btree(p)) {
            postings_parts_[pname] = std::make_unique<PostingsReader>(p);
        }
    }
}

// ── Names ─────────────────────────────────────────────────────────────────────

std::string DiskIndex::get_name(int64_t docid) const {
    if (!names_) throw std::runtime_error("DiskIndex: names part not available");
    return names_->get_name(docid);
}

// ── Lengths ───────────────────────────────────────────────────────────────────

int32_t DiskIndex::get_length(int64_t docid) const {
    if (!lengths_) return 0;
    return lengths_->get_length(docid);
}

LengthStats DiskIndex::get_length_stats(const std::string& field) const {
    if (!lengths_) return {};
    return lengths_->get_stats(field);
}

int64_t DiskIndex::total_documents() const {
    if (!lengths_) return 0;
    return lengths_->total_documents();
}

// ── Postings ──────────────────────────────────────────────────────────────────

bool DiskIndex::has_postings(const std::string& part) const {
    return postings_parts_.count(part) > 0;
}

PostingsReader* DiskIndex::get_postings_reader(const std::string& part) const {
    auto it = postings_parts_.find(part);
    if (it == postings_parts_.end()) return nullptr;
    return it->second.get();
}

std::optional<PostingsIterator>
DiskIndex::get_postings(const std::string& term, const std::string& part) const {
    PostingsReader* pr = get_postings_reader(part);
    if (!pr) return std::nullopt;
    return pr->get_postings(term);
}

} // namespace galago
