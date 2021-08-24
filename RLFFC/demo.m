clear all
clc
warning off;

load('Indian_Pines-40.mat');

H = LP(X1);

numclass = 5;

clusternum = 10 ;

numker = size(H,3);
num = size(H,1);
L_ba = zeros(num,num);

for i = 1 : numker
    L_ba = L_ba + (1 / numker) * H(:,:,i);
end

H0 = zeros(num,num);
qnorm = 2;
opt.disp = 0;
for p=1:numker
    H(:,:,p) = (H(:,:,p)+H(:,:,p)')/2;
    [Hp, ~] = eigs(H(:,:,p), numclass, 'la', opt);
    H0 = H0 + (1/numker)*H(:,:,p);
    HP(:,:,p) = Hp;
end
[Y_ba,~] = eigs(H0, numclass, 'la', opt);
lambda = 2.^-15;
beta = 0.1;


   [F,~,~,~] = RLFFC(HP,numclass,lambda,Y_ba,beta,L_ba);
        for k = 1 : length(clusternum)
            [IDX, C, SUMD, D] = kmeans(F,clusternum(k),'maxiter',100,'replicates',50,'emptyaction','singleton');
            [~,I] = min(D);
        end
