%%%%%Test
C_vec=[0.01;0.03;0.1;0.3;1;3;10;30];
sigma_vec=C_vec;

size_C_Vec=size(C_vec,1);
size_sigma_Vec=size(sigma_vec,1);

%%%Matrix R is of dimensions (size_C_vec*size_sigma_vec) x 3 where the first 2 columns contain
%%%every possible combination of C and sigma values and where the 3rd column contains 
%%%the associated cost for Jval 

%%% C  sigma   %%%Jval
R=zeros(size_C_Vec*size_sigma_Vec,3); 
cnt=0; %%Counter used to control the row index of vector R

for i=1:size_C_Vec
    R(cnt+1:size_C_Vec*i,1)=C_vec(i);
    
  for j=1:size_sigma_Vec
    cnt=cnt+1;
    R(cnt,2)=sigma_vec(j);
  end
end