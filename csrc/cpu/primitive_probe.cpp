// Archive G26/G27/G30-G34/#13-22: real AscendC CPU-debug execution against external independent golden files.
// CPU debugger completion is not NPU or graph acceptance.
#include "tikicpulib.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_store_int2_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool);
extern "C" void oscar_merge_lse_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t);
extern "C" void oscar_status_guard_kernel(uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t);

std::vector<uint8_t> Read(const std::string& path,size_t n) {
    std::vector<uint8_t> data(n);std::ifstream file(path,std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(data.data()),n) || file.peek()!=EOF)
        throw std::runtime_error("golden/input byte count mismatch: "+path);
    return data;
}
struct Gm {
    uint8_t* ptr;size_t size;
    Gm(size_t n,uint8_t fill):ptr(static_cast<uint8_t*>(AscendC::GmAlloc(n))),size(n) {
        if(!ptr) throw std::runtime_error("GmAlloc failed");std::memset(ptr,fill,n);
    }
    Gm(const std::string& path,size_t n):Gm(n,0) {auto v=Read(path,n);std::memcpy(ptr,v.data(),n);}
    ~Gm(){AscendC::GmFree(ptr);}
    Gm(const Gm&)=delete;
};
void Exact(const Gm& output,const std::string& path) {
    auto expected=Read(path,output.size);
    for(size_t i=0;i<output.size;i++) if(output.ptr[i]!=expected[i])
        throw std::runtime_error("exact byte mismatch "+path+" offset="+std::to_string(i)+
           " expected="+std::to_string(expected[i])+" actual="+std::to_string(output.ptr[i]));
}
void Close(const Gm& output,const std::string& path) {
    auto expected=Read(path,output.size);
    for(size_t i=0;i<output.size/4;i++) {
        float a,b;std::memcpy(&a,output.ptr+i*4,4);std::memcpy(&b,expected.data()+i*4,4);
        if(a==b) continue;
        if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>0.005F+0.005F*std::abs(b))
            throw std::runtime_error("float mismatch "+path+" index="+std::to_string(i)+
               " expected="+std::to_string(b)+" actual="+std::to_string(a));
    }
}
int main(int argc,char**argv) {
    try {
        if(argc!=3) throw std::runtime_error("usage: oscar_primitive_cpu store|merge case_directory");
        std::string dir=argv[2],op=argv[1];std::ifstream meta(dir+"/shape.txt");
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if(op=="store") {
            int64_t n,h,d,b,nb,offset,stride,slot64;
            if(!(meta>>n>>h>>d>>b>>nb>>offset>>stride>>slot64)) throw std::runtime_error("bad store shape");
            Gm key(dir+"/key.bin",n*h*d*4),value(dir+"/value.bin",n*h*d*4);
            Gm slots(dir+"/slots.bin",n*(slot64?8:4)),raw(offset+nb*stride,173),status(n*h*4,0x85);
            ICPU_RUN_KF(oscar_store_int2_kernel,4,key.ptr,value.ptr,slots.ptr,raw.ptr,status.ptr,
                        n,h,d,b,nb,offset,stride,slot64);
            Exact(raw,dir+"/expected_raw.bin");Exact(status,dir+"/expected_status.bin");
        } else if(op=="merge") {
            int64_t rows,splits,dim;
            if(!(meta>>rows>>splits>>dim)) throw std::runtime_error("bad merge shape");
            Gm partial(dir+"/partial.bin",rows*splits*dim*4),partialLse(dir+"/partial_lse.bin",rows*splits*4);
            Gm output(rows*dim*4,0x85),lse(rows*4,0x85),status(rows*4,0x85);
            ICPU_RUN_KF(oscar_merge_lse_kernel,4,partial.ptr,partialLse.ptr,output.ptr,lse.ptr,status.ptr,rows,splits,dim);
            Exact(status,dir+"/expected_status.bin");Close(output,dir+"/expected_output.bin");Close(lse,dir+"/expected_lse.bin");
        } else if(op=="guard") {
            int64_t na,nb,nc,nd;
            if(!(meta>>na>>nb>>nc>>nd)) throw std::runtime_error("bad guard shape");
            Gm a(dir+"/a.bin",na*4),b(dir+"/b.bin",nb*4),c(dir+"/c.bin",nc*4),d(dir+"/d.bin",nd*4);
            ICPU_RUN_KF(oscar_status_guard_kernel,4,a.ptr,b.ptr,c.ptr,d.ptr,na,nb,nc,nd);
        } else throw std::runtime_error("unknown op");
        std::cout<<"{\"backend\":\"ascendc_cpu_debug\",\"status\":\"passed\",\"op\":\""<<op<<"\"}"<<std::endl;
        return 0;
    } catch(const std::exception& e) {std::cerr<<"CPU_DEBUG_FAILED: "<<e.what()<<std::endl;return 1;}
}
