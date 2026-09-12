#include "../include/s2_sampler.h"
#include <cassert>
#include <iostream>
#include <vector>
int main(){
 std::vector<float> logits={1,1,1,1,1,1,1,1};
 s2::SamplerParams p; p.temperature=1.0f;p.top_p=1.0f;p.top_k=0;
 s2::SamplerRng a(123456789),b(123456789),c(987654321);
 bool diverged=false;
 for(int i=0;i<1000;++i){int x=s2::sample_token(logits.data(),logits.size(),p,-1,&a);int y=s2::sample_token(logits.data(),logits.size(),p,-1,&b);int z=s2::sample_token(logits.data(),logits.size(),p,-1,&c);assert(x==y);if(x!=z)diverged=true;}
 assert(diverged);
 s2::SamplerParams greedy; greedy.temperature=0;greedy.top_p=1;greedy.top_k=0;
 // Equal-logit greedy tie must choose lowest token ID.
 s2::SamplerRng d(1); assert(s2::sample_token(logits.data(),logits.size(),greedy,-1,&d)==0);
 std::cout<<"SAMPLER_TEST_PASS\n";
}
