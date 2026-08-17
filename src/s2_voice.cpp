// s2_voice.cpp — Voice profile persistence
#include "../include/s2_voice.h"
#include "../third_party/filesystem.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <io.h>
#else
#  include <unistd.h>
#  include <fcntl.h>
#endif

namespace fs = ghc::filesystem;

namespace s2 {

static const char    MAGIC[8] = {'S','2','V','O','I','C','E','\0'};
static const uint32_t VERSION  = 1;

static FILE * open_profile_file(const std::string & path, bool write) {
#ifdef _WIN32
    const std::wstring wp = fs::path(path).wstring();
    return _wfopen(wp.c_str(), write ? L"wb" : L"rb");
#else
    return std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
}

static bool sync_profile_file(FILE * f) {
    if (!f || std::fflush(f) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return ::fsync(fileno(f)) == 0;
#endif
}

static void remove_profile_file(const std::string & path) {
#ifdef _WIN32
    const std::wstring wp = fs::path(path).wstring();
    if (!wp.empty()) DeleteFileW(wp.c_str());
#else
    std::remove(path.c_str());
#endif
}

static bool is_valid_voice_id(const std::string & voice_id) noexcept {
    if (voice_id.empty() || voice_id.size() > 128) return false;
    for (unsigned char c : voice_id) {
        const bool ascii_alnum = (c >= 'a' && c <= 'z') ||
                                 (c >= 'A' && c <= 'Z') ||
                                 (c >= '0' && c <= '9');
        if (!(ascii_alnum || c == '_' || c == '-')) return false;
    }
    return true;
}

bool VoiceProfile::save(const std::string & path) const {
    if (transcript.empty() || transcript.size() >= 1024u * 1024u ||
        num_codebooks <= 0 || T_prompt <= 0 || sample_rate <= 0 || codebook_size <= 0) {
        return false;
    }
    const uint64_t expected_codes = static_cast<uint64_t>(num_codebooks) * static_cast<uint64_t>(T_prompt);
    if (expected_codes != static_cast<uint64_t>(codes.size())) return false;
    for (int32_t code : codes) if (code < 0 || code >= codebook_size) return false;

    // Write beside the destination and replace only after every byte was written,
    // flushed and synced. This preserves the previous valid profile if writing the
    // temporary file fails before replacement.
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
#ifdef _WIN32
    const unsigned long pid = GetCurrentProcessId();
#else
    const long pid = static_cast<long>(::getpid());
#endif
    const std::string tmp_path = path + ".tmp." + std::to_string(pid) + "." +
                                 std::to_string(stamp) + "." + std::to_string(tid);
    struct TempCleanup {
        std::string path;
        ~TempCleanup() { if (!path.empty()) remove_profile_file(path); }
    } cleanup{tmp_path};

    FILE * out = open_profile_file(tmp_path, true);
    if (!out) return false;
    auto close_out = [&]() { const int rc = std::fclose(out); out = nullptr; return rc == 0; };
    auto put = [&](const void * ptr, size_t n) { return n == 0 || std::fwrite(ptr, 1, n, out) == n; };

    const uint32_t version = VERSION;
    const uint64_t transcript_len = static_cast<uint64_t>(transcript.size() + 1);
    const uint64_t codes_size = static_cast<uint64_t>(codes.size()) * sizeof(int32_t);
    bool ok = put(MAGIC, sizeof(MAGIC)) &&
              put(&version, sizeof(version)) &&
              put(&num_codebooks, sizeof(num_codebooks)) &&
              put(&T_prompt, sizeof(T_prompt)) &&
              put(&sample_rate, sizeof(sample_rate)) &&
              put(&codebook_size, sizeof(codebook_size)) &&
              put(&transcript_len, sizeof(transcript_len)) &&
              put(&codes_size, sizeof(codes_size)) &&
              put(transcript.c_str(), static_cast<size_t>(transcript_len)) &&
              put(codes.data(), static_cast<size_t>(codes_size));
    if (ok) ok = sync_profile_file(out);
    if (!close_out()) ok = false;
    if (!ok) return false;

#ifdef _WIN32
    const fs::path tmp_fs(tmp_path);
    const fs::path dst_fs(path);
    if (!MoveFileExW(tmp_fs.wstring().c_str(), dst_fs.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return false;
#else
    // The temp file lives beside the destination, so rename() is atomic with
    // respect to readers on the same filesystem.
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) return false;
    // The file itself was fsync'd before rename. Best-effort fsync of its parent
    // directory makes the rename durable across a sudden power loss on filesystems
    // that require directory metadata to be flushed separately.
    fs::path parent = fs::path(path).parent_path();
    if (parent.empty()) parent = fs::path(".");
    const int dfd = ::open(parent.string().c_str(), O_RDONLY);
    if (dfd >= 0) { (void)::fsync(dfd); ::close(dfd); }
#endif
    cleanup.path.clear();
    return true;
}

VoiceProfile VoiceProfile::load(const std::string & path) {
    FILE * in = open_profile_file(path, false);
    if (!in) throw std::runtime_error("cannot open voice profile: " + path);
    struct FileGuard { FILE * f; ~FileGuard() { if (f) std::fclose(f); } } guard{in};
    auto get = [&](void * ptr, size_t n, const char * what) {
        if (n != 0 && std::fread(ptr, 1, n, in) != n) throw std::runtime_error(std::string("truncated voice profile ") + what);
    };

    char magic[8] = {};
    get(magic, sizeof(magic), "header");
    if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) throw std::runtime_error("invalid voice profile magic");
    uint32_t version = 0;
    get(&version, sizeof(version), "header");
    if (version != VERSION) throw std::runtime_error("unsupported voice profile version");

    VoiceProfile profile;
    get(&profile.num_codebooks, sizeof(profile.num_codebooks), "header");
    get(&profile.T_prompt, sizeof(profile.T_prompt), "header");
    get(&profile.sample_rate, sizeof(profile.sample_rate), "header");
    get(&profile.codebook_size, sizeof(profile.codebook_size), "header");
    uint64_t transcript_len = 0, codes_size = 0;
    get(&transcript_len, sizeof(transcript_len), "header");
    get(&codes_size, sizeof(codes_size), "header");

    constexpr uint64_t MAX_TRANSCRIPT = 1024 * 1024;
    constexpr uint64_t MAX_CODES_BYTES = 512ull * 1024 * 1024;
    if (profile.num_codebooks <= 0 || profile.T_prompt <= 0 || profile.sample_rate <= 0 || profile.codebook_size <= 0)
        throw std::runtime_error("invalid voice profile dimensions");
    if (transcript_len == 0 || transcript_len > MAX_TRANSCRIPT)
        throw std::runtime_error("invalid voice profile transcript length");
    if (codes_size == 0 || codes_size > MAX_CODES_BYTES || codes_size % sizeof(int32_t) != 0)
        throw std::runtime_error("invalid voice profile codes size");
    if (transcript_len > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        codes_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        throw std::runtime_error("voice profile is too large for this platform");

    const uint64_t n_codes_u64 = codes_size / sizeof(int32_t);
    const uint64_t expected_codes = static_cast<uint64_t>(profile.num_codebooks) * static_cast<uint64_t>(profile.T_prompt);
    if (n_codes_u64 != expected_codes)
        throw std::runtime_error("voice profile code count does not match num_codebooks * T_prompt");

    std::vector<char> transcript_buf(static_cast<size_t>(transcript_len));
    get(transcript_buf.data(), transcript_buf.size(), "transcript");
    if (transcript_buf.back() != '\0') throw std::runtime_error("transcript not null-terminated");
    profile.transcript.assign(transcript_buf.data(), transcript_buf.size() - 1);
    if (profile.transcript.empty()) throw std::runtime_error("voice profile transcript is empty");

    profile.codes.resize(static_cast<size_t>(n_codes_u64));
    get(profile.codes.data(), static_cast<size_t>(codes_size), "codes");
    for (int32_t code : profile.codes)
        if (code < 0 || code >= profile.codebook_size)
            throw std::runtime_error("voice profile contains an out-of-range codebook id");
    if (std::fgetc(in) != EOF) throw std::runtime_error("voice profile contains unexpected trailing data");
    if (std::ferror(in)) throw std::runtime_error("failed while reading voice profile");
    return profile;
}

bool VoiceProfile::is_compatible(int32_t expected_num_codebooks,
                                  int32_t expected_codebook_size,
                                  int32_t expected_sample_rate) const {
    return num_codebooks == expected_num_codebooks &&
           codebook_size == expected_codebook_size &&
           sample_rate   == expected_sample_rate;
}

// ---------------------------------------------------------------------------

void VoiceProfileManager::set_storage_dir(const std::string & dir) {
    storage_dir_ = dir;
}

std::string VoiceProfileManager::get_path(const std::string & voice_id) const {
    if (!is_valid_voice_id(voice_id))
        throw std::invalid_argument("voice id must be 1..128 ASCII letters/digits, '_' or '-'");
    fs::path dir(storage_dir_);
    return (dir / (voice_id + ".s2voice")).string();
}

bool VoiceProfileManager::save(const std::string & voice_id, const VoiceProfile & profile) {
    const std::string path = get_path(voice_id); // validates id before touching disk
    fs::path dir(storage_dir_);
    if (!fs::exists(dir)) fs::create_directories(dir);
    if (!fs::is_directory(dir)) return false;
    return profile.save(path);
}

VoiceProfile VoiceProfileManager::load(const std::string & voice_id) {
    return VoiceProfile::load(get_path(voice_id));
}

bool VoiceProfileManager::remove(const std::string & voice_id) {
    const std::string path = get_path(voice_id);
    if (!fs::exists(path)) return false;
    return fs::remove(path);
}

std::vector<std::string> VoiceProfileManager::list() const {
    std::vector<std::string> result;
    if (!fs::exists(storage_dir_)) return result;
    for (const auto & entry : fs::directory_iterator(storage_dir_)) {
        if (fs::is_regular_file(entry.path()) && entry.path().extension() == ".s2voice") {
            const std::string id = entry.path().stem().string();
            // Do not advertise manually-created filenames that the same manager
            // would reject on load/delete because they are not valid voice IDs.
            if (is_valid_voice_id(id)) result.push_back(id);
        }
    }
    return result;
}

} // namespace s2
