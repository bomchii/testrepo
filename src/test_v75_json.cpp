#include "../include/s2_json.h"
#include <cassert>
#include <iostream>
#include <string>
int main(){
 std::string o,e;
 assert(s2::json::normalize_surrogate_pairs(R"({"x":"\uD83D\uDE00","y":"مرحبا","z":"\u4F60"})",o,&e));
 assert(o.find(u8"😀")!=std::string::npos); assert(o.find(u8"مرحبا")!=std::string::npos); assert(o.find("\\u4F60")!=std::string::npos);
 assert(!s2::json::normalize_surrogate_pairs(R"({"x":"\uD83D"})",o,&e));
 assert(!s2::json::normalize_surrogate_pairs(R"({"x":"\uDE00"})",o,&e));
 assert(!s2::json::normalize_surrogate_pairs(R"({"x":"\uDE00\uD83D"})",o,&e));
 assert(!s2::json::normalize_surrogate_pairs(std::string("{\"x\":\"") + char(0xC0)+char(0xAF)+"\"}",o,&e));
 std::cout<<"JSON_SURROGATE_TEST_PASS\n";
}
