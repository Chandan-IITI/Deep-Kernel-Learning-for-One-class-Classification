function [Ktrain, Ktest, A] = kernel_rbf(M_train, M_test, gamma)

N = size(M_train,2);  NN = size(M_test,2);

Dtrain = ((sum(M_train'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,N))'-(2*(M_train'*M_train)));
Dtest = ((sum(M_train'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,N))'-(2*(M_train'*M_test)));

A = sqrt(mean(mean(Dtrain)));

if gamma==0
  Ktrain = exp(-Dtrain/(A*A));          Ktest = exp(-Dtest/(A*A));
else
    Ktrain = exp(-Dtrain/(gamma*gamma));  Ktest = exp(-Dtest/(gamma*gamma));
    A = gamma;
end


