// Archive #126/#145/#150/#151/#154/#164/#172, complete startup D.4.
// Real extracted Physical and packed-loader methods; deferred host DMA model.
// D.4: bounded metadata only, no history expansion, no FP changes, no new UB.
// This checks control/address/byte equivalence, not asynchronous CANN hardware.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>
#define __aicore__
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
enum class TPosition {VECCALC};
enum class HardEvent {MTE2_S,MTE2_V,V_MTE2};
struct Environment {
  std::vector<std::function<void()>> pending;
  int tableDmas=0,scalarReads=0,scalarFences=0;
  void drain(){for(auto& f:pending)f();pending.clear();}
};
thread_local Environment* environment=nullptr;
template<typename T> struct Local {
  std::vector<uint8_t>* bytes;size_t offset=0;
  Local operator[](size_t index)const{return {bytes,offset+index*sizeof(T)};}
  template<typename U> Local<U> ReinterpretCast()const{return {bytes,offset};}
  T GetValue(size_t index)const{
    require(environment->pending.empty(),"scalar UB read before MTE2_S");
    require(offset+(index+1)*sizeof(T)<=bytes->size(),"scalar UB out of bounds");
    T result;std::memcpy(&result,bytes->data()+offset+index*sizeof(T),sizeof(T));
    ++environment->scalarReads;return result;
  }
};
template<TPosition> struct TBuf {
  std::vector<uint8_t> bytes;
  explicit TBuf(size_t size=10240):bytes(size,0xa5){}
  template<typename T> Local<T> Get(){return {&bytes,0};}
};
template<typename T> struct Global {
  std::vector<T>* values;size_t offset=0;
  Global operator[](int64_t index)const{
    require(index>=0,"negative GM offset");return {values,offset+size_t(index)};
  }
};
struct DataCopyExtParams{uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,rsv;};
template<typename T>struct DataCopyPadExtParams{bool enabled;uint8_t left,right;T value;};
template<typename T>void DataCopyPad(Local<T>dst,Global<T>src,DataCopyExtParams p,DataCopyPadExtParams<T>){
  if constexpr(sizeof(T)==4){require(p.blockCount==1&&p.blockLen==4,"metadata DMA changed");++environment->tableDmas;}
  const size_t dstStep=((p.blockLen+31)/32)*32+p.dstStride;
  for(uint32_t row=0;row<p.blockCount;++row){
    const size_t to=dst.offset+row*dstStep,from=src.offset*sizeof(T)+row*(p.blockLen+p.srcStride);
    require(to+p.blockLen<=dst.bytes->size(),"DMA UB overflow");
    require(from+p.blockLen<=src.values->size()*sizeof(T),"DMA GM overflow");
    environment->pending.push_back([=]{std::memcpy(dst.bytes->data()+to,
        reinterpret_cast<const uint8_t*>(src.values->data())+from,p.blockLen);});
  }
}
template<typename T>void Duplicate(Local<T>dst,T value,int count){
  require(environment->pending.empty(),"Vector overwrites outstanding packed DMA");
  for(int i=0;i<count;++i){
    require(dst.offset+(i+1)*sizeof(T)<=dst.bytes->size(),"Vector UB overflow");
    std::memcpy(dst.bytes->data()+dst.offset+i*sizeof(T),&value,sizeof(T));
  }
}
template<HardEvent E>void Fence(){
  if constexpr(E==HardEvent::MTE2_S){++environment->scalarFences;environment->drain();}
  if constexpr(E==HardEvent::MTE2_V)environment->drain();
}
int64_t Min64(int64_t a,int64_t b){return a<b?a:b;}
int64_t Max64(int64_t a,int64_t b){return a>b?a:b;}
struct Geometry{int64_t columns=17,blockTokens=256,blocks=3,ssmOffset=0,pageStride=256*144,hk=1;};
struct Kernel {
  static constexpr int kHistoryHalfKv=32,kPackedStride=160,kSlotBytes=144;
  int kHalfKv;
  Geometry g;int64_t request=0,kvend=0,kvhead=0;int32_t lane=0,error=0,liveKvRows=0;
  bool candidate,decode;Global<int32_t>bt;Global<uint8_t>bytes;
  TBuf<TPosition::VECCALC>addressBuf{128},packedBuf,packedNextBuf;
@@FIELDS@@
  explicit Kernel(bool active,bool isDecode,std::vector<int32_t>&table,std::vector<uint8_t>&raw):
      kHalfKv(isDecode?64:16),candidate(active),decode(isDecode),bt{&table},bytes{&raw}{Init();}
  void Init(){
@@RESET@@
  }
@@ORIGINAL_PHYSICAL@@
@@CACHED_PHYSICAL@@
  bool Physical(int64_t pos,int64_t&block,int64_t&inpage){
    return candidate?PhysicalCached(pos,block,inpage):PhysicalOriginal(pos,block,inpage);
  }
@@DECODE_LOAD@@
@@CLUSTER_LOAD@@
  void load(int64_t start){if(decode)LoadPackedInto(start,packedBuf);else LoadPacked(start);}
};

template<bool Decode>int packed_cases(){
  int count=0;
  for(int lane:{0,1})for(int start:{0,1,16,63,64,77,112,127,128,255})
  for(int tail:{1,17,63,64,129,256})for(int fault:{0,1,2}){
    std::vector<int32_t>table(17*4);for(size_t i=0;i<table.size();++i)table[i]=int32_t((i*5+1)%6);
    if(fault==1)table[(start/128)%17]=-1;
    if(fault==2)table[((start/128)+1)%17]=6;
    std::vector<uint8_t>raw(3*256*144);for(size_t i=0;i<raw.size();++i)raw[i]=uint8_t(i*37+i/144);
    Environment a,b;Kernel old(false,Decode,table,raw),active(true,Decode,table,raw);
    old.lane=active.lane=lane;old.kvend=active.kvend=start+tail;
    old.error=active.error=4;
    for(int iteration=0;iteration<3;++iteration){
      environment=&a;old.load(start);
      environment=&b;active.load(start);
      require(old.packedBuf.bytes==active.packedBuf.bytes,"packed output bytes changed");
      require(old.liveKvRows==active.liveKvRows&&old.error==active.error,"packed live/error changed");
      require(a.scalarFences==b.scalarFences,"incidental packed DMA completion fence removed");
      require(a.pending.empty()&&b.pending.empty(),"packed loader left pending MTE2 work");
    }
    require(b.tableDmas<=a.tableDmas,"cache adds a metadata DMA");++count;
  }
  return count;
}

int physical_cases(){
  int count=0;
  for(int blockTokens:{128,256,2688}){
    std::vector<int32_t>table(17*4);std::vector<uint8_t>raw(1);
    Environment a,b;Kernel old(false,true,table,raw),active(true,true,table,raw);
    old.g.blockTokens=active.g.blockTokens=blockTokens;
    const auto*address=table.data();
    for(int generation=0;generation<3;++generation){
      const int bound=3*(blockTokens/128);
      for(size_t i=0;i<table.size();++i)table[i]=(i%7==0)?-1:(i%11==0)?bound:
          (i%13==0)?std::numeric_limits<int32_t>::max():int32_t((i*7+generation)%bound);
      old.Init();active.Init();
      require(address==table.data(),"table allocation changed between launch generations");
      // Repeated same-page reads, request changes, fragmented mappings,
      // invalid mapping values, invalid positions and fresh graph generations.
      for(int req:{0,0,1,1,0,3,2,2})for(int column=0;column<17;++column)
      for(int offset:{0,0,1,16,63,127,128,-1})for(int prior:{0,2,4,5}){
        old.request=active.request=req;old.error=active.error=prior;
        std::fill(old.addressBuf.bytes.begin(),old.addressBuf.bytes.end(),0x6d);
        std::fill(active.addressBuf.bytes.begin(),active.addressBuf.bytes.end(),0xb7);
        int unrelatedA=0,unrelatedB=0;
        // Model unrelated pending packed/tag MTE2 before Physical. A hit must
        // retain the original completion edge, even though it adds no DMA.
        a.pending.push_back([&]{unrelatedA=17;});b.pending.push_back([&]{unrelatedB=17;});
        const int64_t position=column*128+offset;
        int64_t ba=-19,ia=-23,bb=-19,ib=-23;
        environment=&a;const bool va=old.Physical(position,ba,ia);
        environment=&b;const bool vb=active.Physical(position,bb,ib);
        require(std::tie(va,ba,ia,old.error)==std::tie(vb,bb,ib,active.error),"physical result/error differs");
        require(unrelatedA==unrelatedB&&a.scalarFences==b.scalarFences,"hit changed MTE2 completion boundary");
        a.drain();b.drain();++count;
      }
      // Cached invalid mappings must re-publish error=1 on every call, even
      // after an intervening finite/tag error overwrote the prior error.
      old.request=active.request=0;
      for(int prior:{0,2,4,5}){
        old.error=active.error=prior;int64_t ba=91,ia=93,bb=91,ib=93;
        environment=&a;bool va=old.Physical(0,ba,ia);
        environment=&b;bool vb=active.Physical(0,bb,ib);
        require(!va&&!vb&&old.error==1&&active.error==1&&ba==bb&&ia==ib,"invalid cached value suppressed error");++count;
      }
    }
    // End each launch on the same valid key, then replace that exact table
    // value before Init. Address identity must never confer replay freshness.
    for(int replacement:{1,-1,2,3*(blockTokens/128)}){
      table[5]=replacement;old.Init();active.Init();
      old.request=active.request=0;
      for(int repeat=0;repeat<2;++repeat){
        old.error=active.error=4;
        std::fill(active.addressBuf.bytes.begin(),active.addressBuf.bytes.end(),0x3a);
        int64_t ba=71,ia=73,bb=71,ib=73;
        environment=&a;bool va=old.Physical(5*128+19,ba,ia);
        environment=&b;bool vb=active.Physical(5*128+19,bb,ib);
        require(std::tie(va,ba,ia,old.error)==std::tie(vb,bb,ib,active.error),
            "same-key changed-table graph replay reused stale page mapping");++count;
      }
    }
    require(b.tableDmas<a.tableDmas&&b.scalarReads<a.scalarReads,"repeated metadata lookup did not reduce work");
  }
  return count;
}

int main(){try{
  const int physical=physical_cases(),decode=packed_cases<true>(),cluster=packed_cases<false>();
  std::cout<<"{\"physical_calls\":"<<physical<<",\"decode_packed_cases\":"<<decode
    <<",\"cluster_packed_cases\":"<<cluster
    <<",\"fences\":\"identical\",\"packed_bytes\":\"identical\",\"control\":\"identical\","
      "\"CANN\":\"not_run\",\"NPU\":\"not_run\",\"performance\":\"not_run\"}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
