#include "../include/s2_text.h"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

static void expect_n(const std::string& s, size_t n) {
    auto v=s2::split_text_segments(s,0);
    if(v.size()!=n){
        std::cerr<<"expected "<<n<<" got "<<v.size()<<" for "<<s<<"\n";
        for(auto&x:v)std::cerr<<"["<<x.text<<"]\n";
        std::abort();
    }
}

int main(){
    // Arabic/Persian/Urdu: no ASCII-space requirement and combining marks preserved.
    expect_n(u8"مَرْحَبًا؟كَيْفَ حَالُكَ؟",2);
    expect_n(u8"سلام۔خوبی۔",2);
    expect_n(u8"یہ ٹھیک ہے۔اگلا جملہ۔",2);

    // Indic punctuation shared across scripts.
    expect_n(u8"पहला।दूसरा॥तीसरा।",3);
    expect_n(u8"আমি ভালো।তুমি কেমন।",2);
    // Regression: Telugu letters are visible/spoken text, not Sentence_Break=Extend.
    expect_n(u8"తెలుగు వాక్యం।మరొక వాక్యం।",2);
    expect_n(u8"ಕನ್ನಡ ವಾಕ್ಯ।ಮುಂದಿನದು।",2);
    expect_n(u8"മലയാളം വാക്യം।അടുത്തത്।",2);
    expect_n(u8"ગુજરાતી વાક્ય।બીજું।",2);
    expect_n(u8"ਪੰਜਾਬੀ ਵਾਕ।ਅਗਲਾ।",2);

    // Southeast Asian combining sequences must stay lexical; native terminals split.
    expect_n(u8"សួស្តី។លាហើយ។",2);
    expect_n(u8"မင်္ဂလာပါ။နောက်တစ်ခု။",2);

    // Armenian U+055E is not a universal sentence terminator; U+0589 is.
    expect_n(u8"Ինչպե՞ս ես։ Լավ եմ։",2);

    // Cyrillic initials must not be split at one-letter abbreviations.
    expect_n(u8"А. С. Пушкин пришёл. Потом ушёл.",2);

    // CJK and Hangul: punctuation and tailored ASCII/fullwidth full stops without spaces.
    expect_n(u8"你好！！真的？！再见。",3);
    expect_n(u8"你好.再见.",2);
    expect_n(u8"これは文です．次です．",2);
    expect_n(u8"안녕.다시 만나.",2);

    // Greek question mark is normally encoded as ASCII ';' after NFC. Tailoring is local.
    expect_n(u8"Τι κάνεις;Καλά είμαι.",2);
    expect_n(u8"Τι κάνεις;Καλά είμαι.",2);
    expect_n(u8"Ελληνικά και Latin; this is still one sentence.",1);
    expect_n(u8"Greek Ελληνικά; question. Latin text; not a question.",3);

    // Tibetan shad is a phrase/expression mark, not a reliable sentence boundary in UAX guidance.
    expect_n(u8"བོད་ཡིག། ཚིག་གཞན།",1);

    // Expressive tags are indivisible even when they contain terminal punctuation.
    expect_n(u8"[whisper. softly?] Hello. Next.",2);
    expect_n(u8"[低声说。不要切开！]你好。再见。",2);

    // Speaker tags survive every segment; malformed tags remain literal.
    auto sp=s2::split_text_segments(u8"<|speaker:0|>你好。再见。<|speaker:1|>مرحبا؟وداعا؟",0);
    assert(sp.size()==4); assert(s2::distinct_speaker_count(sp)==2);
    assert(sp[0].text.rfind("<|speaker:0|>",0)==0 && sp[1].text.rfind("<|speaker:0|>",0)==0);
    assert(sp[2].text.rfind("<|speaker:1|>",0)==0 && sp[3].text.rfind("<|speaker:1|>",0)==0);
    auto bad=s2::split_text_segments("<|speaker:x|>Literal. Next.",0);
    assert(bad.size()==2 && s2::distinct_speaker_count(bad)==0 && bad[0].text.find("<|speaker:x|>")!=std::string::npos);
    auto huge=s2::split_text_segments("<|speaker:999999999999999999999999|>Literal. Next.",0);
    assert(huge.size()==2 && s2::distinct_speaker_count(huge)==0);

    // Combining marks/bidi controls must not inflate min-seg-chars enough to prevent merging.
    auto merged=s2::split_text_segments(u8"اَ؟بَ؟",3);
    assert(merged.size()==1);
    // UAX #29 Format controls and soft hyphen are preserved byte-for-byte but
    // must not inflate --min-seg-chars.
    auto controls=s2::split_text_segments(u8"A­​⁧⁩. B.",3);
    assert(controls.size()==1);
    assert(controls[0].text.find(u8"­")!=std::string::npos);
    assert(controls[0].text.find(u8"​")!=std::string::npos);

    // Strict UTF-8 rejection at the splitter boundary.
    std::string invalid="ok"; invalid.push_back(static_cast<char>(0xC0)); invalid.push_back(static_cast<char>(0xAF));
    assert(s2::split_text_segments(invalid,0).empty());


    // Long-form chunking: Unicode-safe, speaker-aware, expressive tags indivisible.
    auto lf=s2::split_text_chunks(u8"<|speaker:0|>你好世界你好世界你好世界。[whisper, slowly!]مرحبا بالعالم مرحبا بالعالم؟",8,2);
    assert(lf.size()>=3);
    for (const auto & x:lf) assert(!x.text.empty());
    for (size_t i=0;i<lf.size();++i) assert(lf[i].text.rfind("<|speaker:0|>",0)==0);
    std::string joined;
    for (const auto & x:lf) joined += x.text.substr(std::string("<|speaker:0|>").size());
    assert(joined==u8"你好世界你好世界你好世界。[whisper, slowly!]مرحبا بالعالم مرحبا بالعالم؟");
    bool tag_whole=false;
    for (const auto & x:lf) if (x.text.find("[whisper, slowly!]")!=std::string::npos) tag_whole=true;
    assert(tag_whole);

    auto emoji=s2::split_text_chunks(u8"A👩‍👩‍👧‍👦B👨‍💻C",2,0);
    assert(!emoji.empty());
    std::string ej; for (const auto & x:emoji) ej+=x.text; assert(ej==u8"A👩‍👩‍👧‍👦B👨‍💻C");
    for (const auto & x:emoji) {
        const std::string zwj = u8"‍";
        assert(x.text.size() < zwj.size() || x.text.compare(x.text.size()-zwj.size(),zwj.size(),zwj)!=0);
    }

    // Hard-splits must not cut immediately after Indic viramas/conjoiners.
    auto indic=s2::split_text_chunks(u8"क्षत्रिय বাংলা తెలుగు",1,0);
    assert(!indic.empty());
    std::string ij; for (const auto & x:indic) ij += x.text;
    assert(ij==u8"क्षत्रिय বাংলা తెలుగు");
    const std::vector<std::string> viramas={u8"्",u8"্",u8"్"};
    for (const auto & x:indic) for (const auto & v:viramas)
        assert(x.text.size()<v.size() || x.text.compare(x.text.size()-v.size(),v.size(),v)!=0);

    // Regional indicators form flag graphemes and stay paired across hard cuts.
    auto flags=s2::split_text_chunks(u8"A🇲🇽B🇯🇵C",1,0);
    assert(!flags.empty());
    std::string fj; for (const auto & x:flags) fj += x.text;
    assert(fj==u8"A🇲🇽B🇯🇵C");
    for (const auto & x:flags) assert(x.text!=u8"🇲" && x.text!=u8"🇽" && x.text!=u8"🇯" && x.text!=u8"🇵");

    auto speakers=s2::split_text_chunks(u8"<|speaker:0|>aaaa bbbb cccc.<|speaker:1|>dddd eeee ffff.",6,0);
    assert(!speakers.empty());
    for (const auto & x:speakers) assert(x.has_speaker);
    for (size_t i=1;i<speakers.size();++i) if (speakers[i-1].speaker_id!=speakers[i].speaker_id) assert(speakers[i].speaker_id==1);

    // Linear-time regression guard: many ambiguous periods/initials.
    std::string perf; perf.reserve(300000); for(int i=0;i<25000;++i) perf += "A. word. ";
    auto t0=std::chrono::steady_clock::now(); auto pv=s2::split_text_segments(perf,0); auto t1=std::chrono::steady_clock::now();
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    assert(pv.size()==25000);
    std::cout<<"TEXT_TEST_PASS segments="<<pv.size()<<" ms="<<ms<<"\n";
}
