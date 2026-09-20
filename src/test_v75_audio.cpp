#include "../include/s2_audio.h"
#include "base64.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

static double zero_crossing_hz(const std::vector<float> & x, int sr) {
    if (x.size() < 3 || sr <= 0) return 0.0;
    size_t begin = x.size() / 8, end = x.size() - x.size() / 8;
    size_t crossings = 0;
    for (size_t i = begin + 1; i < end; ++i) {
        if ((x[i - 1] < 0.0f && x[i] >= 0.0f) || (x[i - 1] > 0.0f && x[i] <= 0.0f)) ++crossings;
    }
    const double seconds = static_cast<double>(end - begin) / sr;
    return seconds > 0.0 ? crossings / (2.0 * seconds) : 0.0;
}

int main(){
 std::vector<float> a(4096,0.125f);
 // Audit13: WSOLA speed changes keep finite audio and approximate duration.
 auto fast=s2::audio_time_stretch(a.data(),a.size(),44100,2.0f);
 auto slow=s2::audio_time_stretch(a.data(),a.size(),44100,0.5f);
 assert(!fast.empty() && fast.size() >= 1800 && fast.size() <= 2300);
 assert(!slow.empty() && slow.size() >= 7800 && slow.size() <= 8300);

 // A speech-length tone should keep approximately the same pitch after WSOLA.
 constexpr int sr=44100;
 constexpr double hz=220.0;
 std::vector<float> tone(sr);
 for(size_t i=0;i<tone.size();++i) tone[i]=0.35f*std::sin(2.0*3.14159265358979323846*hz*i/sr);
 auto tone_fast=s2::audio_time_stretch(tone.data(),tone.size(),sr,1.5f);
 auto tone_slow=s2::audio_time_stretch(tone.data(),tone.size(),sr,0.75f);
 for(const auto * v : {&tone_fast,&tone_slow}) {
   assert(!v->empty());
   const double measured=zero_crossing_hz(*v,sr);
   assert(measured > 205.0 && measured < 235.0);
 }

 s2::audio_normalize_loudness(a,-18.0f);
 for(float v:a) assert(std::isfinite(v) && std::abs(v)<=1.0f);
 assert(s2::audio_write_wav("/mnt/data/v75-audio-test.wav",a.data(),a.size(),44100));
#ifndef _WIN32
 assert(!s2::audio_write_wav("/dev/full",a.data(),a.size(),44100));
#endif

 // Strict/canonical base64 vectors used by inline Fish references[].
 struct B64 { const char * encoded; const char * raw; };
 for (const auto & v : {B64{"", ""}, B64{"Zg==", "f"}, B64{"Zm8=", "fo"},
                        B64{"Zm9v", "foo"}, B64{"SGVsbG8sIHdvcmxkIQ==", "Hello, world!"}}) {
   std::vector<uint8_t> decoded;
   assert(s2::base64_decode(v.encoded, decoded));
   assert(std::string(decoded.begin(), decoded.end()) == v.raw);
   assert(s2::base64_encode(reinterpret_cast<const unsigned char *>(v.raw), std::char_traits<char>::length(v.raw)) == v.encoded);
 }
 for (const char * bad : {"Zg=", "Z===", "Zm9v=", "Zm=v", "!!!!", "Zg==x", "Zh=="}) {
   std::vector<uint8_t> decoded;
   assert(!s2::base64_decode(bad, decoded));
 }

 // Security regression: reject an undersized RIFF fmt chunk before invoking
 // dr_wav. A valid WAV fmt base header is at least 16 bytes.
 const unsigned char bad_fmt_wav[] = {
   'R','I','F','F', 12,0,0,0, 'W','A','V','E',
   'f','m','t',' ', 0,0,0,0
 };
 s2::AudioData malformed;
 assert(!s2::load_audio_from_memory_limited(bad_fmt_wav, sizeof(bad_fmt_wav), malformed, 44100, 30));

 std::cout<<"AUDIO_WRITE_TEST_PASS\n";
}
