#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>
static IMP original;
static FILE *output;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static atomic_int active=0,pending=0,tag=0;
static atomic_ulong seq=0;
static double now(void){static mach_timebase_info_data_t t; if(!t.denom)mach_timebase_info(&t);return (double)mach_absolute_time()*t.numer/t.denom/1e9;}
static void hooked(id object,SEL selector){
 if(atomic_load(&active)){
  unsigned long n=atomic_fetch_add(&seq,1);int request=atomic_load(&tag);double cpu=now();atomic_fetch_add(&pending,1);
  [(id<MTLCommandBuffer>)object addCompletedHandler:^(id<MTLCommandBuffer> cb){
   double completed=now();pthread_mutex_lock(&lock);
   if(output)fprintf(output,"{\"id\":%lu,\"request\":%d,\"submit\":%.9f,\"gpu_start\":%.9f,\"gpu_end\":%.9f,\"callback\":%.9f,\"status\":%lu}\n",n,request,cpu,cb.GPUStartTime,cb.GPUEndTime,completed,(unsigned long)cb.status);
   pthread_mutex_unlock(&lock);atomic_fetch_sub(&pending,1);
  }];
 }
 ((void(*)(id,SEL))original)(object,selector);
}
__attribute__((visibility("default"))) int wam_probe_install(void){
 @autoreleasepool{
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> q=[device newCommandQueue];id<MTLCommandBuffer> cb=[q commandBuffer];Class cls=object_getClass(cb);SEL sel=@selector(commit);Method m=class_getInstanceMethod(cls,sel);if(!m)return -1;
  original=method_getImplementation(m);if(original==(IMP)hooked)return 1;
  class_replaceMethod(cls,sel,(IMP)hooked,method_getTypeEncoding(m));fprintf(stderr,"probe command-buffer class: %s\n",class_getName(cls));return 0;
 }
}
__attribute__((visibility("default"))) int wam_probe_begin(const char *path){output=fopen(path,"wx");if(!output)return -1;atomic_store(&active,1);return 0;}
__attribute__((visibility("default"))) void wam_probe_tag(int t){atomic_store(&tag,t);}
__attribute__((visibility("default"))) int wam_probe_end(void){atomic_store(&active,0);for(int i=0;i<2000&&atomic_load(&pending);i++)usleep(1000);pthread_mutex_lock(&lock);if(output){fclose(output);output=NULL;}pthread_mutex_unlock(&lock);return atomic_load(&pending);}
