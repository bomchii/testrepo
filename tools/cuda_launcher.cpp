#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <shellapi.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "shell32.lib")

namespace fs = std::filesystem;

namespace {
constexpr char kIndexMagic[8]  = {'S','2','C','I','D','X','0','1'};
constexpr char kFooterMagic[8] = {'S','2','C','E','N','D','0','1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kMaxEntries = 16384;

#ifdef S2_RUNTIME_AMD
constexpr wchar_t kRuntimeCoreW[] = L"s2-amd-core.exe";
constexpr char kRuntimeCoreA[] = "s2-amd-core.exe";
constexpr wchar_t kRuntimePrefixW[] = L"amd-";
constexpr wchar_t kRuntimeMutexPrefixW[] = L"Local\\s2.cpp-amd-";
constexpr char kRuntimeLabel[] = "AMD";
constexpr char kLauncherLabel[] = "s2-amd";
#else
constexpr wchar_t kRuntimeCoreW[] = L"s2-cuda-core.exe";
constexpr char kRuntimeCoreA[] = "s2-cuda-core.exe";
constexpr wchar_t kRuntimePrefixW[] = L"cuda-";
constexpr wchar_t kRuntimeMutexPrefixW[] = L"Local\\s2.cpp-cuda-";
constexpr char kRuntimeLabel[] = "CUDA";
constexpr char kLauncherLabel[] = "s2-cuda";
#endif

struct Handle {
    HANDLE h = INVALID_HANDLE_VALUE;
    Handle() = default;
    explicit Handle(HANDLE v) : h(v) {}
    ~Handle() { if (h != INVALID_HANDLE_VALUE && h != nullptr) CloseHandle(h); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : h(o.h) { o.h = INVALID_HANDLE_VALUE; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            if (h != INVALID_HANDLE_VALUE && h != nullptr) CloseHandle(h);
            h = o.h; o.h = INVALID_HANDLE_VALUE;
        }
        return *this;
    }
    HANDLE get() const { return h; }
    HANDLE release() { HANDLE v=h; h=INVALID_HANDLE_VALUE; return v; }
    explicit operator bool() const { return h != INVALID_HANDLE_VALUE && h != nullptr; }
};

struct Mapping {
    Handle file;
    Handle mapping;
    const uint8_t* data = nullptr;
    uint64_t size = 0;
    ~Mapping() { if (data) UnmapViewOfFile(data); }
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& o) noexcept : file(std::move(o.file)), mapping(std::move(o.mapping)), data(o.data), size(o.size) { o.data=nullptr; o.size=0; }
    Mapping& operator=(Mapping&&) = delete;
};

#pragma pack(push,1)
struct Footer {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t index_offset;
    uint64_t index_size;
    uint64_t payload_start;
    uint64_t payload_end;
    uint8_t index_sha256[32];
};
#pragma pack(pop)
static_assert(sizeof(Footer) == 80, "bundle footer layout changed");

struct Entry {
    std::string name;
    uint64_t offset = 0;
    uint64_t size = 0;
    std::array<uint8_t,32> sha{};
};
struct Bundle {
    std::wstring self_path;
    std::array<uint8_t,32> id{};
    uint64_t payload_start = 0;
    uint64_t payload_end = 0;
    std::vector<Entry> entries;
};

[[noreturn]] void win_fail(const char* what) {
    const DWORD e = GetLastError();
    throw std::runtime_error(std::string(what) + " (Win32 " + std::to_string(e) + ")");
}

std::wstring self_path() {
    std::vector<wchar_t> b(1024);
    for (;;) {
        DWORD n=GetModuleFileNameW(nullptr,b.data(),static_cast<DWORD>(b.size()));
        if (!n) win_fail("GetModuleFileNameW failed");
        if (n < b.size()-1) return std::wstring(b.data(), n);
        if (b.size() > 32768) throw std::runtime_error("executable path is too long");
        b.resize(b.size()*2);
    }
}

Mapping map_readonly(const std::wstring& p) {
    Mapping m;
    m.file=Handle(CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!m.file) win_fail("cannot open launcher executable");
    LARGE_INTEGER n{};
    if (!GetFileSizeEx(m.file.get(), &n) || n.QuadPart <= 0) win_fail("GetFileSizeEx failed");
    m.size=static_cast<uint64_t>(n.QuadPart);
    if (m.size > static_cast<uint64_t>((std::numeric_limits<SIZE_T>::max)()))
        throw std::runtime_error("launcher executable is too large to map");
    m.mapping=Handle(CreateFileMappingW(m.file.get(),nullptr,PAGE_READONLY,0,0,nullptr));
    if (!m.mapping) win_fail("CreateFileMappingW failed");
    m.data=static_cast<const uint8_t*>(MapViewOfFile(m.mapping.get(),FILE_MAP_READ,0,0,0));
    if (!m.data) win_fail("MapViewOfFile failed");
    return m;
}

std::array<uint8_t,32> sha256_bytes(const uint8_t* data, size_t size) {
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    if (BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0) < 0)
        throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA256) failed");
    struct AlgClose { BCRYPT_ALG_HANDLE h; ~AlgClose(){ if(h) BCryptCloseAlgorithmProvider(h,0); } } ac{alg};
    DWORD objlen=0, cb=0, hashlen=0;
    if (BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&objlen),sizeof(objlen),&cb,0)<0 ||
        BCryptGetProperty(alg,BCRYPT_HASH_LENGTH,reinterpret_cast<PUCHAR>(&hashlen),sizeof(hashlen),&cb,0)<0 || hashlen!=32)
        throw std::runtime_error("BCryptGetProperty(SHA256) failed");
    std::vector<uint8_t> obj(objlen);
    if (BCryptCreateHash(alg,&hash,obj.data(),objlen,nullptr,0,0)<0)
        throw std::runtime_error("BCryptCreateHash failed");
    struct HashClose { BCRYPT_HASH_HANDLE h; ~HashClose(){ if(h) BCryptDestroyHash(h); } } hc{hash};
    size_t off=0;
    while (off<size) {
        const ULONG chunk=static_cast<ULONG>((std::min)(size-off,static_cast<size_t>(0x7fffffffu)));
        if (BCryptHashData(hash,const_cast<PUCHAR>(data+off),chunk,0)<0)
            throw std::runtime_error("BCryptHashData failed");
        off+=chunk;
    }
    std::array<uint8_t,32> out{};
    if (BCryptFinishHash(hash,out.data(),static_cast<ULONG>(out.size()),0)<0)
        throw std::runtime_error("BCryptFinishHash failed");
    return out;
}

std::array<uint8_t,32> sha256_file(const fs::path& p) {
    Handle f(CreateFileW(p.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN,nullptr));
    if (!f) win_fail("cannot open extracted runtime file for hashing");
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    if (BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0) throw std::runtime_error("SHA256 init failed");
    struct A{BCRYPT_ALG_HANDLE h;~A(){if(h)BCryptCloseAlgorithmProvider(h,0);}} a{alg};
    DWORD objlen=0,cb=0;
    if (BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&objlen),sizeof(objlen),&cb,0)<0) throw std::runtime_error("SHA256 property failed");
    std::vector<uint8_t> obj(objlen);
    if (BCryptCreateHash(alg,&hash,obj.data(),objlen,nullptr,0,0)<0) throw std::runtime_error("SHA256 create failed");
    struct H{BCRYPT_HASH_HANDLE h;~H(){if(h)BCryptDestroyHash(h);}} hh{hash};
    std::array<uint8_t,1<<20> buf{};
    for (;;) {
        DWORD got=0;
        if (!ReadFile(f.get(),buf.data(),static_cast<DWORD>(buf.size()),&got,nullptr)) win_fail("ReadFile while hashing failed");
        if (!got) break;
        if (BCryptHashData(hash,buf.data(),got,0)<0) throw std::runtime_error("SHA256 update failed");
    }
    std::array<uint8_t,32> out{};
    if (BCryptFinishHash(hash,out.data(),32,0)<0) throw std::runtime_error("SHA256 finish failed");
    return out;
}

std::string hex32(const std::array<uint8_t,32>& a) {
    static constexpr char h[]="0123456789abcdef";
    std::string s; s.resize(64);
    for(size_t i=0;i<a.size();++i){s[2*i]=h[a[i]>>4];s[2*i+1]=h[a[i]&15];}
    return s;
}
std::wstring whex32(const std::array<uint8_t,32>& a){auto s=hex32(a);return std::wstring(s.begin(),s.end());}

uint16_t rd16(const uint8_t*& p,const uint8_t* e){if(e-p<2)throw std::runtime_error("truncated bundle index");uint16_t v=p[0]|uint16_t(p[1])<<8;p+=2;return v;}
uint32_t rd32(const uint8_t*& p,const uint8_t* e){if(e-p<4)throw std::runtime_error("truncated bundle index");uint32_t v=0;for(int i=0;i<4;++i)v|=uint32_t(p[i])<<(8*i);p+=4;return v;}
uint64_t rd64(const uint8_t*& p,const uint8_t* e){if(e-p<8)throw std::runtime_error("truncated bundle index");uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(p[i])<<(8*i);p+=8;return v;}

std::string lower_ascii(std::string s){for(char&c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
bool valid_name(const std::string& n){
    if(n.empty()||n.size()>1024||n.front()=='/'||n.back()=='/')return false;
    static constexpr const char * kInvalid="<>:\"\\|?*";
    static const std::set<std::string> dev={"con","prn","aux","nul","clock$","com1","com2","com3","com4","com5","com6","com7","com8","com9","lpt1","lpt2","lpt3","lpt4","lpt5","lpt6","lpt7","lpt8","lpt9"};
    size_t pos=0;
    while(pos<n.size()){
        size_t slash=n.find('/',pos);
        std::string part=n.substr(pos,slash==std::string::npos?std::string::npos:slash-pos);
        if(part.empty()||part=="."||part==".."||part.size()>240||part.back()=='.'||part.back()==' ')return false;
        for(unsigned char c:part){if(c<0x20||c>0x7e||std::strchr(kInvalid,static_cast<int>(c))!=nullptr)return false;}
        auto dot=part.find('.');auto stem=part.substr(0,dot);while(!stem.empty()&&(stem.back()=='.'||stem.back()==' '))stem.pop_back();
        stem=lower_ascii(std::move(stem));if(dev.find(stem)!=dev.end())return false;
        if(slash==std::string::npos)break;pos=slash+1;
    }
    return true;
}

Bundle parse_bundle() {
    Bundle b; b.self_path=self_path();
    Mapping m=map_readonly(b.self_path);
    if(m.size<sizeof(Footer))throw std::runtime_error("CUDA payload footer is missing");
    bool found=false; Footer f{}; uint64_t footer_pos=0;
    for(uint64_t pos=m.size-sizeof(Footer);;--pos){
        if(std::memcmp(m.data+pos,kFooterMagic,8)==0){
            Footer cand{}; std::memcpy(&cand,m.data+pos,sizeof(cand));
            bool bounds=cand.version==kVersion && cand.reserved==0 &&
                cand.index_offset<=pos && cand.index_size<=pos-cand.index_offset &&
                cand.index_offset+cand.index_size==pos &&
                cand.payload_start<=cand.payload_end && cand.payload_end==cand.index_offset;
            if(bounds){
                auto ih=sha256_bytes(m.data+cand.index_offset,static_cast<size_t>(cand.index_size));
                if(std::memcmp(ih.data(),cand.index_sha256,32)==0){f=cand;footer_pos=pos;found=true;break;}
            }
        }
        if(pos==0)break;
    }
    (void)footer_pos;
    if(!found)throw std::runtime_error("valid CUDA payload footer/index was not found");
    std::copy(std::begin(f.index_sha256),std::end(f.index_sha256),b.id.begin());
    b.payload_start=f.payload_start;b.payload_end=f.payload_end;
    const uint8_t* p=m.data+f.index_offset; const uint8_t* e=p+f.index_size;
    if(e-p<24||std::memcmp(p,kIndexMagic,8)!=0)throw std::runtime_error("invalid CUDA bundle index magic"); p+=8;
    uint32_t ver=rd32(p,e), count=rd32(p,e); uint64_t ps=rd64(p,e);
    if(ver!=kVersion||count==0||count>kMaxEntries||ps!=b.payload_start)throw std::runtime_error("invalid CUDA bundle index header");
    std::set<std::string> names; bool core=false;
    for(uint32_t i=0;i<count;++i){
        uint16_t nl=rd16(p,e), flags=rd16(p,e); uint64_t off=rd64(p,e), sz=rd64(p,e);
        if(flags!=0||nl==0||nl>1024||e-p<32+nl)throw std::runtime_error("invalid CUDA bundle entry");
        Entry x; x.offset=off;x.size=sz;std::copy(p,p+32,x.sha.begin());p+=32;x.name.assign(reinterpret_cast<const char*>(p),nl);p+=nl;
        if(!valid_name(x.name))throw std::runtime_error("unsafe CUDA bundle filename");
        auto key=lower_ascii(x.name);if(!names.insert(key).second)throw std::runtime_error("duplicate CUDA bundle filename");
        if(key==lower_ascii(kRuntimeCoreA))core=true;
        if(off<b.payload_start||off>b.payload_end||sz>b.payload_end-off)throw std::runtime_error("CUDA bundle entry range is out of bounds");
        auto actual=sha256_bytes(m.data+off,static_cast<size_t>(sz));if(actual!=x.sha)throw std::runtime_error("CUDA bundle payload hash mismatch: "+x.name);
        b.entries.push_back(std::move(x));
    }
    if(p!=e||!core)throw std::runtime_error("CUDA bundle index has trailing data or no core executable");
    auto byoff=b.entries;std::sort(byoff.begin(),byoff.end(),[](const Entry&a,const Entry&c){return a.offset<c.offset;});
    uint64_t end=b.payload_start;for(const auto&x:byoff){if(x.offset!=end)throw std::runtime_error("CUDA payload ranges are overlapping or contain an unindexed gap");end=x.offset+x.size;}
    if(end!=b.payload_end)throw std::runtime_error("CUDA payload range does not exactly cover the indexed payload");
    return b;
}

fs::path runtime_root(){
    DWORD n=GetEnvironmentVariableW(L"LOCALAPPDATA",nullptr,0);if(!n)win_fail("LOCALAPPDATA is not defined");
    std::vector<wchar_t>b(n);if(!GetEnvironmentVariableW(L"LOCALAPPDATA",b.data(),n))win_fail("cannot read LOCALAPPDATA");
    return fs::path(b.data())/L"s2.cpp"/L"runtime";
}
fs::path cache_path(const Bundle&b){return runtime_root()/(std::wstring(kRuntimePrefixW)+whex32(b.id));}

Handle lock_mutex_for(const std::wstring& id){
    std::wstring name=std::wstring(kRuntimeMutexPrefixW)+id;Handle h(CreateMutexW(nullptr,FALSE,name.c_str()));if(!h)win_fail("CreateMutexW failed");
    DWORD r=WaitForSingleObject(h.get(),INFINITE);if(r!=WAIT_OBJECT_0&&r!=WAIT_ABANDONED)win_fail("WaitForSingleObject failed");return h;
}
void unlock_mutex(Handle&h){if(h&& !ReleaseMutex(h.get()))win_fail("ReleaseMutex failed");}

bool cache_active(const fs::path& cache){
    fs::path marker=cache/L".active.lock";
    if(!fs::exists(marker))return false;
    Handle h(CreateFileW(marker.c_str(),DELETE|GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
    if(h)return false;
    DWORD e=GetLastError();return e==ERROR_SHARING_VIOLATION||e==ERROR_LOCK_VIOLATION||e==ERROR_ACCESS_DENIED;
}

bool read_complete(const fs::path& p,const std::string&id){
    std::ifstream f(p, std::ios::binary);if(!f)return false;std::string s((std::istreambuf_iterator<char>(f)),{});return s==id+"\n";
}

std::wstring lower_ascii_w(std::wstring s){
    for(wchar_t&c:s)if(c>=L'A'&&c<=L'Z')c=static_cast<wchar_t>(c-L'A'+L'a');
    return s;
}

bool plain_file(const fs::path&p){
    DWORD a=GetFileAttributesW(p.c_str());
    return a!=INVALID_FILE_ATTRIBUTES && !(a&FILE_ATTRIBUTE_DIRECTORY) && !(a&FILE_ATTRIBUTE_REPARSE_POINT);
}

bool plain_directory(const fs::path&p){
    DWORD a=GetFileAttributesW(p.c_str());
    return a!=INVALID_FILE_ATTRIBUTES && (a&FILE_ATTRIBUTE_DIRECTORY) && !(a&FILE_ATTRIBUTE_REPARSE_POINT);
}

bool verify_cache(const Bundle&b,const fs::path& cache){
    try {
        if(!plain_directory(cache))return false;

        // The DLL loader searches the executable directory. Hashing only the
        // manifest entries is not sufficient if an unindexed DLL can be planted
        // beside the core. Require exact directory contents and reject reparse
        // points so the verified cache is also the cache that is executed.
        std::set<std::wstring> allowed={L".complete",L".active.lock"};
        for(const auto&x:b.entries){std::wstring wn(x.name.begin(),x.name.end());std::replace(wn.begin(),wn.end(),L'/',fs::path::preferred_separator);allowed.insert(lower_ascii_w(wn));}
        std::error_code ec;
        for(const auto&de:fs::recursive_directory_iterator(cache,fs::directory_options::none,ec)){
            if(ec)return false;
            const fs::path p=de.path();
            if(de.is_directory(ec)){if(ec||!plain_directory(p))return false;continue;}
            if(!plain_file(p))return false;
            fs::path rel=fs::relative(p,cache,ec);if(ec)return false;
            if(allowed.find(lower_ascii_w(rel.wstring()))==allowed.end())return false;
        }
        if(ec)return false;

        const fs::path complete=cache/L".complete";
        if(!plain_file(complete)||!read_complete(complete,hex32(b.id)))return false;
        for(const auto&x:b.entries){
            std::wstring wn(x.name.begin(),x.name.end());std::replace(wn.begin(),wn.end(),L'/',fs::path::preferred_separator);fs::path p=cache/fs::path(wn);
            if(!plain_file(p))return false;
            auto sz=fs::file_size(p,ec);if(ec||sz!=x.size)return false;if(sha256_file(p)!=x.sha)return false;
        }
        const fs::path active=cache/L".active.lock";
        if(fs::exists(active,ec)&&(!plain_file(active)||ec))return false;
        return !ec;
    } catch (...) {
        return false;
    }
}

void copy_payload_file(HANDLE self,const Entry&x,const fs::path&dst){
    Handle out(CreateFileW(dst.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr));if(!out)win_fail("cannot create extracted CUDA runtime file");
    LARGE_INTEGER li{};li.QuadPart=static_cast<LONGLONG>(x.offset);if(!SetFilePointerEx(self,li,nullptr,FILE_BEGIN))win_fail("SetFilePointerEx failed");
    uint64_t left=x.size;std::vector<uint8_t>buf(1<<20);
    while(left){DWORD want=static_cast<DWORD>((std::min<uint64_t>)(left,buf.size())),got=0;if(!ReadFile(self,buf.data(),want,&got,nullptr)||got!=want)win_fail("reading embedded CUDA payload failed");
        DWORD put=0;if(!WriteFile(out.get(),buf.data(),got,&put,nullptr)||put!=got)win_fail("writing CUDA runtime cache failed");left-=got;}
    if(!FlushFileBuffers(out.get()))win_fail("FlushFileBuffers for CUDA runtime failed");
}

void ensure_cache(const Bundle&b,const fs::path&cache){
    fs::create_directories(runtime_root());
    if(verify_cache(b,cache))return;
    if(fs::exists(cache)){
        if(cache_active(cache))throw std::runtime_error("runtime cache is active but corrupt; refusing to modify it");
        std::error_code ec;fs::remove_all(cache,ec);if(ec)throw std::runtime_error("cannot remove corrupt runtime cache");
    }
    fs::path tmp=fs::path(cache.wstring()+L".tmp-"+std::to_wstring(GetCurrentProcessId())+L"-"+std::to_wstring(GetTickCount64()));
    std::error_code ec;fs::remove_all(tmp,ec);ec.clear();fs::create_directories(tmp,ec);if(ec)throw std::runtime_error("cannot create temporary runtime directory");
    try{
        Handle self(CreateFileW(b.self_path.c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));if(!self)win_fail("cannot reopen launcher payload");
        for(const auto&x:b.entries){std::wstring wn(x.name.begin(),x.name.end());std::replace(wn.begin(),wn.end(),L'/',fs::path::preferred_separator);fs::path d=tmp/fs::path(wn);std::error_code dec;fs::create_directories(d.parent_path(),dec);if(dec)throw std::runtime_error("cannot create runtime payload directory");copy_payload_file(self.get(),x,d);if(sha256_file(d)!=x.sha)throw std::runtime_error("extracted runtime hash mismatch: "+x.name);}
        {std::ofstream f(tmp/L".complete",std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot write runtime completion marker");f<<hex32(b.id)<<"\n";f.flush();if(!f)throw std::runtime_error("cannot flush runtime completion marker");}
        if(!MoveFileExW(tmp.c_str(),cache.c_str(),MOVEFILE_WRITE_THROUGH))win_fail("atomic runtime cache rename failed");
    }catch(...){std::error_code ignored;fs::remove_all(tmp,ignored);throw;}
    if(!verify_cache(b,cache))throw std::runtime_error("runtime cache verification failed after extraction");
}

Handle active_marker(const fs::path&cache){
    SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};
    Handle h(CreateFileW((cache/L".active.lock").c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_ALWAYS,FILE_ATTRIBUTE_HIDDEN,nullptr));if(!h)win_fail("cannot create runtime active marker");
    DWORD flags=0;if(!GetHandleInformation(h.get(),&flags)||!(flags&HANDLE_FLAG_INHERIT))throw std::runtime_error("runtime active marker is not inheritable");return h;
}

std::wstring quote_arg(const std::wstring&s){if(s.empty())return L"\"\"";if(s.find_first_of(L" \t\"")==std::wstring::npos)return s;std::wstring o=L"\"";size_t sl=0;for(wchar_t c:s){if(c==L'\\'){++sl;continue;}if(c==L'\"'){o.append(sl*2+1,L'\\');o+=L'\"';sl=0;continue;}o.append(sl,L'\\');sl=0;o+=c;}o.append(sl*2,L'\\');o+=L'\"';return o;}

DWORD launch_core(const fs::path&core,int argc,wchar_t**argv,HANDLE active){
    std::wstring cmd=quote_arg(core.wstring());for(int i=1;i<argc;++i){cmd+=L" ";cmd+=quote_arg(argv[i]);}
    STARTUPINFOEXW sx{};sx.StartupInfo.cb=sizeof(sx);sx.StartupInfo.dwFlags=STARTF_USESTDHANDLES;
    std::vector<Handle> dup;std::vector<HANDLE> inherit;inherit.push_back(active);
    auto dupstd=[&](DWORD which,DWORD nul_access)->HANDLE{
        HANDLE src=GetStdHandle(which);HANDLE d=nullptr;
        if(src&&src!=INVALID_HANDLE_VALUE){
            if(!DuplicateHandle(GetCurrentProcess(),src,GetCurrentProcess(),&d,0,TRUE,DUPLICATE_SAME_ACCESS))win_fail("DuplicateHandle(stdio) failed");
        }else{
            SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};
            d=CreateFileW(L"NUL",nul_access,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
            if(!d||d==INVALID_HANDLE_VALUE)win_fail("opening NUL fallback for stdio failed");
        }
        dup.emplace_back(d);inherit.push_back(d);return d;
    };
    sx.StartupInfo.hStdInput=dupstd(STD_INPUT_HANDLE,GENERIC_READ);
    sx.StartupInfo.hStdOutput=dupstd(STD_OUTPUT_HANDLE,GENERIC_WRITE);
    sx.StartupInfo.hStdError=dupstd(STD_ERROR_HANDLE,GENERIC_WRITE);
    SIZE_T bytes=0;InitializeProcThreadAttributeList(nullptr,1,0,&bytes);std::vector<uint8_t>storage(bytes);sx.lpAttributeList=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if(!InitializeProcThreadAttributeList(sx.lpAttributeList,1,0,&bytes))win_fail("InitializeProcThreadAttributeList failed");
    struct Attr{LPPROC_THREAD_ATTRIBUTE_LIST p;~Attr(){if(p)DeleteProcThreadAttributeList(p);}}attr{sx.lpAttributeList};
    if(!UpdateProcThreadAttribute(sx.lpAttributeList,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,inherit.data(),inherit.size()*sizeof(HANDLE),nullptr,nullptr))win_fail("UpdateProcThreadAttribute(handle list) failed");
    PROCESS_INFORMATION pi{};std::vector<wchar_t>mutable_cmd(cmd.begin(),cmd.end());mutable_cmd.push_back(L'\0');
    if(!CreateProcessW(core.c_str(),mutable_cmd.data(),nullptr,nullptr,TRUE,EXTENDED_STARTUPINFO_PRESENT,nullptr,nullptr,&sx.StartupInfo,&pi))win_fail("CreateProcessW(runtime core) failed");
    Handle ph(pi.hProcess),th(pi.hThread);DWORD w=WaitForSingleObject(ph.get(),INFINITE);if(w!=WAIT_OBJECT_0)win_fail("waiting for CUDA core failed");DWORD ec=1;if(!GetExitCodeProcess(ph.get(),&ec))win_fail("GetExitCodeProcess failed");return ec;
}

int runtime_info(const Bundle&b){auto c=cache_path(b);std::wcout<<L"bundle_sha256: "<<whex32(b.id)<<L"\ncache: "<<c.wstring()<<L"\n";std::cout<<"payload_files: "<<b.entries.size()<<"\n";for(const auto&x:b.entries)std::cout<<"  "<<x.name<<" size="<<x.size<<" sha256="<<hex32(x.sha)<<"\n";std::cout<<"cache_valid: "<<(verify_cache(b,c)?"yes":"no")<<"\ncache_active: "<<(cache_active(c)?"yes":"no")<<"\n";return 0;}

bool hash_suffix(const std::wstring&name,std::wstring&out){const std::wstring prefix(kRuntimePrefixW);if(name.rfind(prefix,0)!=0||name.size()<prefix.size()+64)return false;out=name.substr(prefix.size(),64);if(out.size()!=64)return false;for(auto c:out)if(!((c>=L'0'&&c<=L'9')||(c>=L'a'&&c<=L'f')||(c>=L'A'&&c<=L'F')))return false;std::transform(out.begin(),out.end(),out.begin(),::towlower);return true;}
int clean_runtime(){fs::path root=runtime_root();if(!fs::exists(root)){std::cout<<"runtime cache is already empty\n";return 0;}Handle global=lock_mutex_for(L"clean-runtime");size_t removed=0,skipped=0;for(const auto&de:fs::directory_iterator(root)){if(!de.is_directory())continue;std::wstring id;if(!hash_suffix(de.path().filename().wstring(),id))continue;Handle m=lock_mutex_for(id);fs::path cache=root/(std::wstring(kRuntimePrefixW)+id);if(cache_active(cache)){++skipped;unlock_mutex(m);continue;}std::error_code ec;fs::remove_all(de.path(),ec);if(ec){unlock_mutex(m);unlock_mutex(global);throw std::runtime_error("failed to remove CUDA runtime cache");}++removed;unlock_mutex(m);}unlock_mutex(global);std::cout<<"removed="<<removed<<" active_skipped="<<skipped<<"\n";return 0;}
} // namespace

int wmain(int argc,wchar_t**argv){
    try{
        Bundle b=parse_bundle();
        if(argc==2&&std::wcscmp(argv[1],L"--runtime-info")==0)return runtime_info(b);
        if(argc==2&&std::wcscmp(argv[1],L"--clean-runtime")==0)return clean_runtime();
        fs::path cache=cache_path(b);Handle mutex=lock_mutex_for(whex32(b.id));ensure_cache(b,cache);Handle active=active_marker(cache);unlock_mutex(mutex);fs::path core=cache/fs::path(kRuntimeCoreW);DWORD ec=launch_core(core,argc,argv,active.get());return static_cast<int>(ec);
    }catch(const std::exception&e){std::cerr<<kLauncherLabel<<" launcher: "<<e.what()<<"\n";return 111;}
}
#else
int main(){return 1;}
#endif
