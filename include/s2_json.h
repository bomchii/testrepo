#pragma once
#include <string>

namespace s2 {
namespace json {
// Rewrites only valid UTF-16 surrogate pairs inside JSON string escapes to
// their scalar UTF-8 representation. Rejects isolated/reversed surrogates,
// malformed \u escapes, and invalid raw UTF-8. Non-surrogate bytes/escapes are
// preserved byte-for-byte.
bool normalize_surrogate_pairs(const std::string & input,
                               std::string & output,
                               std::string * error = nullptr);
} // namespace json
} // namespace s2
