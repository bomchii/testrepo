#include "../include/s2_text.h"
#include "s2_utf8.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace s2 {
namespace {
#include "s2_unicode_ranges.inc"
#include "s2_sentence_break_ignore.inc"

template <size_t N>
bool cp_in_ranges(uint32_t cp, const UnicodeRange (&ranges)[N]) noexcept {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo) hi = mid;
        else if (cp > ranges[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t cp) noexcept {
    return cp_in_ranges(cp, kLetterRanges);
}

bool is_greek(uint32_t cp) noexcept {
    return (cp >= 0x0370u && cp <= 0x03FFu) || (cp >= 0x1F00u && cp <= 0x1FFFu);
}

// Unicode 18 Sentence_Break=STerm ranges. Keeping these as data rather than a
// language-specific switch makes sentence punctuation work for Arabic/Indic,
// CJK and other scripts without transliteration or normalization.
static constexpr UnicodeRange kSentenceTerminalRanges[] = {
    {0x0021u,0x0021u},{0x003Fu,0x003Fu},{0x0589u,0x0589u},{0x061Du,0x061Fu},
    {0x06D4u,0x06D4u},{0x0700u,0x0702u},{0x07F9u,0x07F9u},{0x0837u,0x0837u},
    {0x0839u,0x0839u},{0x083Du,0x083Eu},{0x0964u,0x0965u},{0x104Au,0x104Bu},
    {0x1362u,0x1362u},{0x1367u,0x1368u},{0x166Eu,0x166Eu},{0x1735u,0x1736u},
    {0x17D4u,0x17D5u},{0x1803u,0x1803u},{0x1809u,0x1809u},{0x1944u,0x1945u},
    {0x1AA8u,0x1AABu},{0x1B4Eu,0x1B4Fu},{0x1B5Au,0x1B5Bu},{0x1B5Eu,0x1B5Fu},
    {0x1B7Du,0x1B7Fu},{0x1C3Bu,0x1C3Cu},{0x1C7Eu,0x1C7Fu},{0x203Cu,0x203Du},
    {0x2047u,0x2049u},{0x2CF9u,0x2CFBu},{0x2E2Eu,0x2E2Eu},{0x2E3Cu,0x2E3Cu},
    {0x2E53u,0x2E54u},{0x2E60u,0x2E61u},{0x3002u,0x3002u},{0xA4FFu,0xA4FFu},
    {0xA60Eu,0xA60Fu},{0xA6F3u,0xA6F3u},{0xA6F7u,0xA6F7u},{0xA876u,0xA877u},
    {0xA8CEu,0xA8CFu},{0xA92Fu,0xA92Fu},{0xA9C8u,0xA9C9u},{0xAA5Du,0xAA5Fu},
    {0xAAF0u,0xAAF1u},{0xABEBu,0xABEBu},{0xFE12u,0xFE12u},{0xFE15u,0xFE16u},
    {0xFE56u,0xFE57u},{0xFF01u,0xFF01u},{0xFF1Fu,0xFF1Fu},{0xFF61u,0xFF61u},
    {0x10A56u,0x10A57u},{0x10F55u,0x10F59u},{0x10F86u,0x10F89u},
    {0x11047u,0x11048u},{0x110BEu,0x110C1u},{0x11141u,0x11143u},
    {0x111C5u,0x111C6u},{0x111CDu,0x111CDu},{0x111DEu,0x111DFu},
    {0x11238u,0x11239u},{0x1123Bu,0x1123Cu},{0x112A9u,0x112A9u},
    {0x113D4u,0x113D5u},{0x1144Bu,0x1144Cu},{0x115C2u,0x115C3u},
    {0x115C9u,0x115D7u},{0x11641u,0x11642u},{0x1173Cu,0x1173Eu},
    {0x11944u,0x11944u},{0x11946u,0x11946u},{0x11A42u,0x11A43u},
    {0x11A9Bu,0x11A9Cu},{0x11C41u,0x11C42u},{0x11EF7u,0x11EF8u},
    {0x11F43u,0x11F44u},{0x16A6Eu,0x16A6Fu},{0x16AF5u,0x16AF5u},
    {0x16B37u,0x16B38u},{0x16B44u,0x16B44u},{0x16D6Eu,0x16D6Fu},
    {0x16E98u,0x16E98u},{0x1BC9Fu,0x1BC9Fu},{0x1DA88u,0x1DA88u},
};

bool is_aterm(uint32_t cp) noexcept {
    // Unicode 18 Sentence_Break=ATerm.
    return cp == 0x002Eu || cp == 0x2024u || cp == 0xFE52u || cp == 0xFF0Eu;
}

bool is_sentence_terminal(uint32_t cp) noexcept {
    return cp_in_ranges(cp, kSentenceTerminalRanges);
}

bool is_tts_ellipsis(uint32_t cp) noexcept {
    // UAX #29 does not classify ellipsis as STerm, but V7.4 treated it as a
    // useful TTS phrase/sentence boundary. Preserve that behavior explicitly.
    return cp == 0x2026u || cp == 0xFE19u;
}

bool is_noncounting(uint32_t cp) noexcept {
    // UAX #29 SB5 ignores Sentence_Break=Extend and Format when locating
    // sentence boundaries. Use an exact data table instead of hand-written
    // script blocks: broad approximations can accidentally classify visible
    // letters as combining marks (notably Telugu U+0C05..).
    return cp_in_ranges(cp, kSentenceIgnoreRanges);
}

// Conservative no-split linkers for TTS hard chunking.  UAX #29 GB9c uses
// Indic_Conjunct_Break=Linker for the scripts where a virama/conjoiner binds
// the following consonant into the same extended grapheme cluster.  Include
// the common Brahmic viramas as well: keeping a little more text together is
// harmless for a chunk target, while cutting immediately after a virama can
// change shaping/pronunciation.
bool is_grapheme_linker(uint32_t cp) noexcept {
    switch (cp) {
        case 0x094Du: case 0x09CDu: case 0x0A4Du: case 0x0ACDu:
        case 0x0B4Du: case 0x0BCDu: case 0x0C4Du: case 0x0CCDu:
        case 0x0D4Du: case 0x0DCAu: case 0x0E3Au: case 0x0F84u:
        case 0x1039u: case 0x1714u: case 0x1734u: case 0x17D2u:
        case 0x1B44u: case 0xA9C0u: case 0x11046u: case 0x1134Du:
        case 0x113D0u: case 0x11F42u:
            return true;
        default:
            return false;
    }
}

bool is_regional_indicator(uint32_t cp) noexcept {
    return cp >= 0x1F1E6u && cp <= 0x1F1FFu;
}

size_t visible_count(const std::string & s) noexcept {
    size_t n = 0;
    for (size_t p = 0; p < s.size();) {
        uint32_t cp = 0; size_t w = 0;
        if (!utf8::decode_one(s, p, cp, w)) return 0;
        if (!is_noncounting(cp)) ++n;
        p += w;
    }
    return n;
}

bool is_closer(uint32_t cp) noexcept {
    return cp == '"' || cp == '\'' || cp == ')' || cp == ']' || cp == '}' ||
           cp == 0x00BBu || cp == 0x0F3Bu || cp == 0x0F3Du ||
           cp == 0x2019u || cp == 0x201Du || cp == 0x203Au || cp == 0x2046u ||
           cp == 0x2309u || cp == 0x230Bu || cp == 0x232Au ||
           (cp >= 0x2769u && cp <= 0x2775u && (cp & 1u) == 1u) ||
           cp == 0x27C6u || cp == 0x27E7u || cp == 0x27E9u || cp == 0x27EBu ||
           cp == 0x27EDu || cp == 0x27EFu ||
           cp == 0x2984u || cp == 0x2986u || cp == 0x2988u || cp == 0x298Au ||
           cp == 0x298Cu || cp == 0x298Eu || cp == 0x2990u || cp == 0x2992u ||
           cp == 0x2994u || cp == 0x2996u || cp == 0x2998u ||
           cp == 0x29D9u || cp == 0x29DBu || cp == 0x29FDu ||
           cp == 0x2E23u || cp == 0x2E25u || cp == 0x2E27u || cp == 0x2E29u ||
           cp == 0x3009u || cp == 0x300Bu || cp == 0x300Du || cp == 0x300Fu ||
           cp == 0x3011u || cp == 0x3015u || cp == 0x3017u || cp == 0x3019u ||
           cp == 0x301Bu || cp == 0x301Eu || cp == 0x301Fu ||
           cp == 0xFD3Fu || cp == 0xFE18u || cp == 0xFE36u || cp == 0xFE38u ||
           cp == 0xFE3Au || cp == 0xFE3Cu || cp == 0xFE3Eu || cp == 0xFE40u ||
           cp == 0xFE42u || cp == 0xFE44u || cp == 0xFE48u ||
           cp == 0xFE5Au || cp == 0xFE5Cu || cp == 0xFE5Eu ||
           cp == 0xFF09u || cp == 0xFF3Du || cp == 0xFF5Du || cp == 0xFF60u || cp == 0xFF63u;
}

bool is_wrapper(uint32_t cp) noexcept {
    return is_closer(cp) || cp == '(' || cp == '[' || cp == '{' ||
           cp == 0x00ABu || cp == 0x0F3Au || cp == 0x0F3Cu ||
           cp == 0x2018u || cp == 0x201Cu || cp == 0x2039u || cp == 0x2045u ||
           cp == 0x3008u || cp == 0x300Au || cp == 0x300Cu || cp == 0x300Eu ||
           cp == 0x3010u || cp == 0x3014u || cp == 0x3016u || cp == 0x3018u ||
           cp == 0x301Au || cp == 0xFF08u || cp == 0xFF3Bu || cp == 0xFF5Bu;
}

bool is_cjk_no_space_script(uint32_t cp) noexcept {
    return (cp >= 0x1100u && cp <= 0x11FFu) || // Hangul Jamo
           (cp >= 0x2E80u && cp <= 0x2FFFu) || // CJK radicals/IDC
           (cp >= 0x3040u && cp <= 0x30FFu) || // Hiragana/Katakana
           (cp >= 0x3100u && cp <= 0x312Fu) || (cp >= 0x31A0u && cp <= 0x31BFu) ||
           (cp >= 0x31F0u && cp <= 0x31FFu) ||
           (cp >= 0x3400u && cp <= 0x4DBFu) || (cp >= 0x4E00u && cp <= 0x9FFFu) ||
           (cp >= 0xAC00u && cp <= 0xD7AFu) || (cp >= 0xF900u && cp <= 0xFAFFu) ||
           (cp >= 0xFF66u && cp <= 0xFF9Du) ||
           (cp >= 0x20000u && cp <= 0x323AFu);
}

bool is_effective_term(uint32_t cp, bool greek_question_context) noexcept {
    if (cp == ';' || cp == 0x037Eu) return greek_question_context;
    return is_aterm(cp) || is_sentence_terminal(cp) || is_tts_ellipsis(cp);
}

bool has_spoken(const std::string & s) noexcept {
    bool word_has_greek = false;
    for (size_t p = 0; p < s.size();) {
        uint32_t cp=0; size_t w=0;
        if (!utf8::decode_one(s,p,cp,w)) return false;
        if (is_letter(cp)) word_has_greek = is_greek(cp);
        const bool greek_q = (cp == ';' || cp == 0x037Eu) && word_has_greek;
        if (!utf8::is_whitespace(cp) && !is_wrapper(cp) &&
            !is_effective_term(cp, greek_q) && !is_noncounting(cp)) return true;
        if (!is_letter(cp) && !is_noncounting(cp)) word_has_greek = false;
        p += w;
    }
    return false;
}

void unicode_trim(const std::string & s, std::string & lead, std::string & body, std::string & trail) {
    size_t first=s.size(), last=0;
    for (size_t p=0;p<s.size();) {
        uint32_t cp=0; size_t w=0; utf8::decode_one(s,p,cp,w);
        if (!utf8::is_whitespace(cp)) { if (first==s.size()) first=p; last=p+w; }
        p+=w;
    }
    if (first==s.size()) { lead=s; body.clear(); trail.clear(); return; }
    lead.assign(s,0,first); body.assign(s,first,last-first); trail.assign(s,last,s.size()-last);
}

bool parse_speaker_tag(const std::string & s, size_t p, size_t & end, int32_t & id) noexcept {
    static const std::string pre = "<|speaker:";
    if (p + pre.size() > s.size() || s.compare(p, pre.size(), pre) != 0) return false;
    size_t q = p + pre.size();
    if (q >= s.size() || !std::isdigit(static_cast<unsigned char>(s[q]))) return false;
    uint64_t v=0; size_t digits=0;
    while (q < s.size() && std::isdigit(static_cast<unsigned char>(s[q]))) {
        const unsigned digit = static_cast<unsigned>(s[q]-'0');
        if (v > (static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) - digit) / 10u) return false;
        v = v*10u + digit;
        ++q; ++digits;
    }
    if (digits==0 || q+2>s.size() || s[q]!='|' || s[q+1]!='>') return false;
    id=static_cast<int32_t>(v); end=q+2; return true;
}

std::vector<std::pair<size_t,size_t>> balanced_expressive_ranges(const std::string & s) {
    std::vector<std::pair<size_t,size_t>> out;
    std::vector<size_t> stack;
    for (size_t p=0;p<s.size();) {
        uint32_t cp=0; size_t w=0; utf8::decode_one(s,p,cp,w);
        if (cp=='[') stack.push_back(p);
        else if (cp==']' && !stack.empty()) {
            const size_t begin=stack.back(); stack.pop_back();
            if (stack.empty()) out.emplace_back(begin,p+w);
        }
        p+=w;
    }
    return out;
}

bool ascii_abbrev(const std::vector<uint32_t> & word) {
    static const char * const names[] = {"mr","mrs","ms","dr","prof","sr","sra","dra","ing","lic",
        "etc","vs","fig","dept","approx","jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec","ene","abr","ago","dic"};
    if (word.size()==1 && is_letter(word[0])) return true; // Unicode initials, e.g. А. С.
    std::string w;
    for (uint32_t cp:word) {
        if (cp>0x7Fu || !std::isalpha(static_cast<unsigned char>(cp))) return false;
        w.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(cp))));
    }
    for (const char * x:names) if (w==x) return true;
    return false;
}

bool next_starts_cjk(const std::string & text, size_t p) noexcept {
    while (p < text.size()) {
        uint32_t cp=0; size_t w=0;
        if (!utf8::decode_one(text,p,cp,w)) return false;
        if (is_noncounting(cp)) { p += w; continue; }
        return is_cjk_no_space_script(cp);
    }
    return false;
}

std::vector<std::string> split_plain(const std::string & text, int32_t min_chars) {
    std::vector<std::string> out;
    if (text.empty()) return out;
    const auto protected_ranges = balanced_expressive_ranges(text);
    size_t range_i=0;
    std::string cur, pending, pending_lead;
    std::vector<uint32_t> word;
    bool word_has_greek=false;

    auto flush=[&]() {
        std::string lead,body,trail; unicode_trim(cur,lead,body,trail); cur.clear();
        if (body.empty()) { if (!lead.empty()) pending += lead; return; }
        if (!pending.empty()) {
            pending += lead; pending += body; body.swap(pending);
            lead.swap(pending_lead); pending.clear(); pending_lead.clear();
        }
        const bool short_seg = min_chars>0 && visible_count(body)<static_cast<size_t>(min_chars);
        if (!has_spoken(body) || short_seg) {
            if (pending.empty()) pending_lead=std::move(lead); else pending += lead;
            pending += body; pending += trail;
        } else {
            out.push_back(std::move(body));
            if (!trail.empty()) pending_lead=std::move(trail);
        }
        word.clear(); word_has_greek=false;
    };

    for (size_t i=0;i<text.size();) {
        if (range_i<protected_ranges.size() && i==protected_ranges[range_i].first) {
            const size_t e=protected_ranges[range_i].second;
            cur.append(text,i,e-i);
            // Expressive instructions are indivisible. Do not let punctuation
            // inside [ ... ] alter the lexical state surrounding the tag.
            word.clear(); word_has_greek=false;
            i=e; ++range_i; continue;
        }

        uint32_t cp=0; size_t width=0; utf8::decode_one(text,i,cp,width);
        const size_t cp_pos=i;
        cur.append(text,i,width); i+=width;

        if (is_letter(cp)) {
            word.push_back(cp);
            if (is_greek(cp)) word_has_greek=true;
            continue;
        }
        if (is_noncounting(cp)) continue;

        const bool greek_question = (cp == ';' || cp == 0x037Eu) && word_has_greek;
        if (!is_effective_term(cp, greek_question)) {
            word.clear(); word_has_greek=false;
            continue;
        }

        bool ascii_ellipsis=false;
        if (cp=='.') {
            size_t e=i;
            while(e<text.size() && text[e]=='.') ++e;
            if (e-cp_pos>=3) { cur.append(text,i,e-i); i=e; ascii_ellipsis=true; }
        }

        const bool ambiguous_period = is_aterm(cp) && !ascii_ellipsis;
        if (!ambiguous_period) {
            while(i<text.size()) {
                uint32_t x=0;size_t w=0;utf8::decode_one(text,i,x,w);
                if (!is_effective_term(x, greek_question)) break;
                cur.append(text,i,w);i+=w;
            }
        }

        // UAX #29 includes closing punctuation in the sentence before the break.
        while(i<text.size()) {
            uint32_t x=0;size_t w=0;utf8::decode_one(text,i,x,w);
            if (is_noncounting(x)) { cur.append(text,i,w); i+=w; continue; }
            if (!is_closer(x)) break;
            cur.append(text,i,w); i+=w;
        }

        if (cp=='.' && !ascii_ellipsis && ascii_abbrev(word)) {
            word.clear(); word_has_greek=false;
            continue;
        }

        bool boundary = !ambiguous_period || ascii_ellipsis || i>=text.size();
        if (!boundary) {
            uint32_t x=0;size_t w=0;utf8::decode_one(text,i,x,w);
            // Preserve V7.4's conservative full-stop behavior for Latin-like
            // text, but also support CJK/Hangul with no ASCII spaces as UAX #29
            // explicitly recommends tailoring for such scripts.
            boundary = utf8::is_whitespace(x) || next_starts_cjk(text,i);
        }
        if (boundary) flush();
        word.clear(); word_has_greek=false;
    }
    flush();
    if (!pending.empty()) {
        if (!out.empty()) { out.back() += pending_lead; out.back() += pending; }
        else if (has_spoken(pending)) out.push_back(std::move(pending));
    }
    return out;
}
} // namespace

bool contains_valid_speaker_tag(const std::string & text) noexcept {
    for (size_t p = 0; p < text.size(); ++p) {
        if (text[p] != '<') continue;
        size_t end = 0;
        int32_t id = -1;
        if (parse_speaker_tag(text, p, end, id)) return true;
    }
    return false;
}

std::vector<TextSegment> split_text_segments(const std::string & text, int32_t min_chars) {
    std::vector<TextSegment> result;
    if (text.empty() || min_chars < 0 || !utf8::is_valid(text)) return result;
    struct Block { std::string tag; std::string content; int32_t id=-1; bool speaker=false; };
    std::vector<Block> blocks;
    size_t content_start=0, search=0;
    std::string current_tag; int32_t current_id=-1; bool current_speaker=false;
    while (search<text.size()) {
        const size_t p=text.find("<|speaker:",search);
        if (p==std::string::npos) break;
        size_t end=0;int32_t id=-1;
        if (!parse_speaker_tag(text,p,end,id)) { search=p+1; continue; }
        if (p>content_start || !current_tag.empty())
            blocks.push_back({current_tag,text.substr(content_start,p-content_start),current_id,current_speaker});
        current_tag=text.substr(p,end-p); current_id=id; current_speaker=true;
        content_start=end; search=end;
    }
    if (content_start<text.size() || !current_tag.empty() || blocks.empty())
        blocks.push_back({current_tag,text.substr(content_start),current_id,current_speaker});

    for (const auto & b:blocks) {
        auto pieces=split_plain(b.content,min_chars);
        for (auto & piece:pieces) {
            TextSegment seg;
            seg.speaker_id=b.id; seg.has_speaker=b.speaker;
            seg.text.reserve(b.tag.size()+piece.size()); seg.text=b.tag; seg.text+=piece;
            result.push_back(std::move(seg));
        }
    }
    return result;
}


static std::vector<std::string> hard_split_piece(const std::string & piece,
                                                 int32_t target_chars) {
    std::vector<std::string> out;
    if (piece.empty() || target_chars <= 0) return out;
    const auto protected_ranges = balanced_expressive_ranges(piece);
    size_t range_i = 0;
    size_t start = 0;
    size_t p = 0;
    size_t visible = 0;
    size_t last_soft = std::string::npos;

    auto in_protected = [&](size_t pos) {
        while (range_i < protected_ranges.size() && pos >= protected_ranges[range_i].second) ++range_i;
        return range_i < protected_ranges.size() &&
               pos >= protected_ranges[range_i].first && pos < protected_ranges[range_i].second;
    };
    auto emit = [&](size_t end) {
        if (end <= start) return;
        out.emplace_back(piece.substr(start, end - start));
        start = end;
        visible = 0;
        last_soft = std::string::npos;
    };

    while (p < piece.size()) {
        uint32_t cp = 0; size_t w = 0;
        if (!utf8::decode_one(piece, p, cp, w)) return {};
        const size_t next = p + w;
        const bool protected_here = in_protected(p);
        if (!is_noncounting(cp)) ++visible;
        if (!protected_here && utf8::is_whitespace(cp)) last_soft = next;

        if (!protected_here && visible >= static_cast<size_t>(target_chars)) {
            size_t cut = last_soft != std::string::npos && last_soft > start ? last_soft : next;
            // Never leave a combining/format code point at the start of the next
            // chunk and never split immediately after a joiner.
            size_t q = cut;
            bool linker_pending = false;
            bool regional_pending = (cut == next && is_regional_indicator(cp));
            while (q < piece.size()) {
                uint32_t ncp = 0; size_t nw = 0;
                if (!utf8::decode_one(piece, q, ncp, nw)) return {};
                if (is_noncounting(ncp)) {
                    if (ncp == 0x200Du || ncp == 0x200Cu || is_grapheme_linker(ncp)) {
                        linker_pending = true;
                    }
                    q += nw;
                    continue;
                }
                if (linker_pending) {
                    // Joiners/viramas bind the following base into the same
                    // shaping/grapheme sequence. Include that base, then keep
                    // scanning its combining marks and any subsequent linker.
                    q += nw;
                    linker_pending = false;
                    regional_pending = false;
                    continue;
                }
                if (regional_pending && is_regional_indicator(ncp)) {
                    // Emoji flags are pairs of Regional Indicator code points.
                    q += nw;
                    regional_pending = false;
                    continue;
                }
                break;
            }
            if (q > cut && q <= piece.size()) cut = q;
            emit(cut);
            p = cut;
            continue;
        }
        p = next;
    }
    if (start < piece.size()) out.emplace_back(piece.substr(start));
    return out;
}

std::vector<TextSegment> split_text_chunks(const std::string & text,
                                           int32_t target_chars,
                                           int32_t min_chunk_chars) {
    std::vector<TextSegment> out;
    if (text.empty() || target_chars <= 0 || min_chunk_chars < 0 ||
        min_chunk_chars > target_chars || !utf8::is_valid(text)) return out;

    auto sentences = split_text_segments(text, 0);
    for (const auto & sentence : sentences) {
        std::string prefix;
        std::string content = sentence.text;
        if (sentence.has_speaker) {
            prefix = "<|speaker:" + std::to_string(sentence.speaker_id) + "|>";
            if (content.compare(0, prefix.size(), prefix) == 0) content.erase(0, prefix.size());
        }
        auto pieces = hard_split_piece(content, target_chars);
        if (pieces.empty() && !content.empty()) return {};
        for (auto & piece : pieces) {
            TextSegment cand;
            cand.has_speaker = sentence.has_speaker;
            cand.speaker_id = sentence.speaker_id;
            cand.text = prefix + piece;
            const size_t cand_vis = visible_count(piece);
            if (!out.empty() && out.back().has_speaker == cand.has_speaker &&
                (!cand.has_speaker || out.back().speaker_id == cand.speaker_id)) {
                std::string prev_content = out.back().text;
                if (out.back().has_speaker) {
                    const std::string pp = "<|speaker:" + std::to_string(out.back().speaker_id) + "|>";
                    if (prev_content.compare(0, pp.size(), pp) == 0) prev_content.erase(0, pp.size());
                }
                const size_t prev_vis = visible_count(prev_content);
                if (prev_vis + cand_vis <= static_cast<size_t>(target_chars)) {
                    out.back().text += piece;
                    continue;
                }
            }
            out.push_back(std::move(cand));
        }
    }

    if (min_chunk_chars > 0 && out.size() > 1) {
        for (size_t i = out.size(); i-- > 1;) {
            std::string content = out[i].text;
            std::string prefix;
            if (out[i].has_speaker) {
                prefix = "<|speaker:" + std::to_string(out[i].speaker_id) + "|>";
                if (content.compare(0, prefix.size(), prefix) == 0) content.erase(0, prefix.size());
            }
            if (visible_count(content) >= static_cast<size_t>(min_chunk_chars)) continue;
            if (out[i-1].has_speaker != out[i].has_speaker ||
                (out[i].has_speaker && out[i-1].speaker_id != out[i].speaker_id)) continue;
            out[i-1].text += content;
            out.erase(out.begin() + static_cast<std::ptrdiff_t>(i));
        }
    }
    return out;
}

size_t distinct_speaker_count(const std::vector<TextSegment> & segments) {
    std::unordered_set<int32_t> ids;
    for (const auto & s:segments) if (s.has_speaker) ids.insert(s.speaker_id);
    return ids.size();
}
} // namespace s2
