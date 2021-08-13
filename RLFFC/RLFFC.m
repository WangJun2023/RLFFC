function [F,R,gamma,obj] = RLFFC(YP,k,lambda,F_ba,beta,L_ba)

num = size(YP, 1); %the number of samples
numker = size(YP, 3); %m represents the number of kernels
maxIter = 100; %the number of iterations
gamma = ones(numker,1)/(numker);
R = zeros(k,k,numker);

for p=1:numker % m - kernels
    R(:,:,p) = eye(k);
end

Y_ba = zeros(num,k); % k - clusters, N - samples
[F,~] = qr(randn(num,k),0);
opts = [];  opts.info = 1;
opts.gtol = 1e-5;

flag = 1;
iter = 0;
while flag
    iter = iter +1;
    for p=1:numker
        Y_ba = Y_ba + gamma(p)*(YP(:,:,p)*R(:,:,p));
    end
    X = F;
    A = - beta * L_ba;
    G = - Y_ba - lambda * F_ba;
    [F,~] = FOForth(X,G,@fun,opts,A,G); %calculate the minimum
   
    for p=1:numker
        if gamma(p)>1e-4
            TP = gamma(p)*YP(:,:,p)'*(F);
            [Up,Sp,Vp] = svd(TP,'econ');
            R(:,:,p) = Up*Vp';
        end
    end 
    coef = zeros(1,numker);
    for p=1:numker
        coef(1,p) = trace((F)'*YP(:,:,p)* R(:,:,p));
    end
    gamma = coef/norm(coef,2); 
    temp = zeros(num,k);
    for p=1:numker
        temp = temp + gamma(p)*(YP(:,:,p)*R(:,:,p));
    end
    obj(iter) = trace(F'*temp + lambda * F' * F_ba + beta * F' * L_ba * F);
    
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        flag =0;
    end
end

F = F./ repmat(sqrt(sum(F.^2, 2)), 1,k);

    function [funX, F] = fun(X,A,G)
        F = A * X + G;
        funX = sum(sum(X.* F));
    end

end
