#include <mlx/mlx.h>
#include <chrono>
#include <iostream>
namespace mx=mlx::core;
int main(){
 std::string root="records/audit4_20261105/export/";
 auto d=mx::load_safetensors(root+"inputs.safetensors").first;
 mx::Args a; for(auto k:{"x","part_0","part_1","part_2","residual","gate","bias"}) a.push_back(d.at(k));
 mx::eval(a); auto f=mx::import_function(root+"real_projection.mlxfn");
 for(int i=0;i<20;i++){auto y=f(a);mx::eval(y);}
 std::cout<<"[";
 for(int i=0;i<200;i++){auto t=std::chrono::steady_clock::now();auto y=f(a);mx::eval(y);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();if(i)std::cout<<",";std::cout<<ms; if(i==199)mx::save(root+"cpp_output.npy",y[0]);}
 std::cout<<"]";
}
