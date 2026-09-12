#include "../include/s2_audio.h"
#include <cassert>
#include <iostream>
#include <vector>
int main(){
 std::vector<float> a(4096,0.125f);
 assert(s2::audio_write_wav("/mnt/data/v75-audio-test.wav",a.data(),a.size(),44100));
#ifndef _WIN32
 assert(!s2::audio_write_wav("/dev/full",a.data(),a.size(),44100));
#endif
 std::cout<<"AUDIO_WRITE_TEST_PASS\n";
}
