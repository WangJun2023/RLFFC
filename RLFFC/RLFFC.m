function [F,R,gamma,obj] = RLFFC(YP,k,lambda,F_ba,beta,L_ba)

num = size(YP, 1); % the number of bands
view_num = size(YP, 3); % the number of superpixels
maxIter = 100; % the number of iterations

%% Initialization gamma, R, Y, F
gamma = ones(view_num,1)/(view_num);

R = zeros(k,k,view_num);
for v = 1 : view_num
    R(:,:,v) = eye(k);
end

Y_ba = zeros(num,k); 

[F,~] = qr(randn(num,k),0);

opts = [];  opts.info = 1;
opts.gtol = 1e-5;

flag = 1;
iter = 0;

%% Iterative Update
while flag
    iter = iter + 1;
    for v = 1 : view_num
        Y_ba = Y_ba + gamma(v)*(YP(:,:,v)*R(:,:,v));
    end
    
    %% update F
    X = F;
    A = - beta * L_ba;
    G = - Y_ba - lambda * F_ba;
    [F,~] = FOForth(X,G,@fun,opts,A,G); %calculate the minimum
   
   %% update R
    for v = 1 : view_num
        if gamma(v)>1e-4
            temp_matrix = gamma(v)*YP(:,:,v)'*(F);
            [Up,Sp,Vp] = svd(temp_matrix,'econ');
            R(:,:,v) = Up*Vp';
        end
    end
    
    %% update gamma
    coef = zeros(1,view_num);
    for v = 1 : view_num
        coef(1,v) = trace((F)'*YP(:,:,v)* R(:,:,v));
    end
    gamma = coef/norm(coef,2); 
    
    %% calculate objective function value
    temp = zeros(num,k);
    for v = 1 : view_num
        temp = temp + gamma(v)*(YP(:,:,v)*R(:,:,v));
    end
    obj(iter) = trace(F'*temp + lambda * F' * F_ba + beta * F' * L_ba * F);
   
    %% verify convergence
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        flag =0;
    end
end

    F = F./ repmat(sqrt(sum(F.^2, 2)), 1, k);

    function [funX, F] = fun(X,A,G)
        F = A * X + G;
        funX = sum(sum(X.* F));
    end

end
