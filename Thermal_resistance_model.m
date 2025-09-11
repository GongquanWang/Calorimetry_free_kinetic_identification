%This solver aims to generate dataset of kinetics parameters of TR model for features and calculated temperature for input.
clear;
t_cal = 1800; % calculating duration
dt = 0.0005; % Iteration step size
step = 2000; %output step size

cell_type = 1; % 0 for cylindrical battery; 1 for prismatic battery
if cell_type == 1
    Lx = 0.050; Ly = 0.160; Lz = 0.118;  %thickness; length; height; 
    kappa_x= 0.8;  kappa_y= 20; kappa_z= 20; % thermal conductivity of jelly roll
    V = Lx*Ly*Lz;
    Sxy = Lx*Ly; Sxz = Lx*Lz; Syz = Ly*Lz;
    S_heat = Syz; % heating area
else
    r = 0.009; L = 0.065; % radius; height
    kappa_r= 0.8;  kappa_z= 20;% thermal conductivity of jelly roll
    V = 3.14*r^2*L;
    S_top = 3.14*r^2;
    S_heat = 0.03*0.062; % heating area
    S_side = 2*3.14*r*L-S_heat; % heat dissipation area
end

rho = 2200; Cp = 1100;% density; heat capacity;
Ta = 300; % ambient temperature;
R = 8.314;% molar constant;

%% Sampling
dimensions = 15; %number of undefined kinetics parameters (Features);
sample_size= 30; %Sampling frequency
T_num = 100;
sample_num = T_num*sample_size; % number of samples;
Sample = zeros(sample_num, dimensions); % Dataset for features;

T_onset = linspace(366, 550, T_num);
% Onset temperature ranges from experiment statistics are divided into different ranges and then sampled to cover possible maximal onset temperatures.
% 366K is the min onset temperature from experiment statistics; 550K is the max onset temperature;
for i = 1:T_num
    T_range = linspace(343, T_onset(i), 6);%Divide temperature range from SEI reaction to TR onset into 5 sub intervals
    delta_T = (T_onset(i)-343);
    dr_l = 1e-4/delta_T;
    dr_h = 75e-4/delta_T;
    Tpeak_l = 570;
    Tpeak_h = 1250;
    s_min = [T_range(1) T_range(2) T_range(3) T_range(4) T_range(5) Tpeak_l dr_l dr_l dr_l dr_l dr_l 5e6   5   0  5];
    s_max = [T_range(2) T_range(3) T_range(4) T_range(5) T_range(6) Tpeak_h dr_h dr_h dr_h dr_h 8e-4 7e7   100 1  50];
    % Undefined kinetics parameters are:
    % Tonset_1; Tonset_2; Tonset_3; Tonset_4; Tonset_5;Tpeak;
    % dr1; dr2; dr3; dr4; dr5;
    % Flux; h; epsilon;kappa;
    sample_lhs =lhsdesign(sample_size, dimensions); % Latin hypercube sampling
    sample_range=repmat(s_max-s_min,sample_size,1);
    sample_min = repmat(s_min,sample_size,1);
    sample_part =(sample_lhs .* sample_range)+ sample_min;
    Sample((i-1)*sample_size+1:i*sample_size,:) = sample_part;
end

Ti_onset = Sample(:, 1:5); % onset temperatures;
dr = Sample(:, 7:11); % slopes of reaction rate at onset temperatures;
H = zeros(sample_num, 5);
for i =1:5
    H(:, i) = (Sample(:, i+1)-Sample(:, i))*Cp*rho;
end
Flux = Sample(:, 12); Q_heat = S_heat*Flux; % heating flux from external;
h = Sample(:, 13); % convective heat transfer coefficient on cell surface;
epsilon = Sample(:, 14); % emissivity on cell surface;
lambda =  Sample(:, 15);

%% Calculate A and Ea according to dr and T_onset;
A = zeros(sample_num,5);
Ea = zeros(sample_num,5);
for i = 1:sample_num
    for j = 1:5
        fun = @(x) [x(1) - x(2)/(R*Ti_onset(i,j))-log(1e-3);
            x(1) - x(2)/(R*Ti_onset(i,j))+log(x(2))-log(R*Ti_onset(i,j)^2)-log(dr(i,j))];
        x0 = [0, 1e4];
        sol = fsolve(fun, x0);
        A(i,j) = exp(sol(1));
        Ea(i,j) = sol(2);
    end
end

% Thermal runaway model to calculate temperature based on kinetics parameters；

output_T = zeros(t_cal/(dt*step),sample_num);
output_Q = zeros(t_cal/(dt*step),sample_num);

for n = 1:sample_num

    c = ones(1,5); Q = zeros(1,5);
    Tc = 300; % Core temperature of battery
    Ts = 300; % Surface temperature of battery
    Tf = 300; % Surface temperature of steel fixture

    for N = 1:1:(t_cal/dt)

        if (Tc > 550)
            Q_heat(n) = 0;
        end

        for i=1:1:5

            c(i) = c(i)/( 1+A(n,i)*exp( -1*Ea(n,i)/(R*Tc) )*dt );
            Q(i) = A(n,i)*exp( -1*Ea(n,i)/(R*Tc) )*c(i)*H(n,i);
        end

        if cell_type == 1 % prismatic battery         
            R_x = Lx/2/(kappa_x*Syz);
            R_cod = 1/(lambda(n)*Syz);
            R_hx = 1/( h(n)*Syz + epsilon(n)*5.67e-8*(Ta^2+Tf^2)*(Ta+Tf)*Syz );

            R_y = Ly/2/(kappa_y*Sxz);
            R_hy = 1/( h(n)*Sxz + epsilon(n)*5.67e-8*(Ta^2+Ts^2)*(Ta+Ts)*Sxz );

            R_z = Lz/2/(kappa_z*Sxy);
            R_hz = 1/( h(n)*Sxy + epsilon(n)*5.67e-8*(Ta^2+Ts^2)*(Ta+Ts)*Sxy );

            Q_dis = ( (Ta - Tc)/(R_x+R_cod+R_hx)+ 2*(Ta - Tc)/(R_y+R_hy) + 2*(Ta - Tc)/(R_z+R_hz) )/V;

        else % cylindrical battery
            R_r = r/(kappa_r*S_side);
            R_hr = 1/( h(n)*S_side + epsilon(n)*5.67e-8*(Ta^2+Ts^2)*(Ta+Ts)*S_side );

            R_z = L/2/(kappa_z*S_top);
            R_hz = 1/( h(n)*S_top + epsilon(n)*5.67e-8*(Ta^2+Ts^2)*(Ta+Ts)*S_top );

            Q_dis = ( (Ta - Tc)/(R_r+R_hr)+ 2*(Ta - Tc)/(R_z+R_hz) )/V;
        end

        dTc = (sum(Q(:)) + Q_dis + Q_heat(n))/(rho*Cp);
        Tc = Tc + dTc*dt;

        if cell_type == 1
            Ts = (R_x*Ta + R_cod*Tc+R_hx*Tc)/(R_x+R_cod+R_hx);
            Tf = (R_x*Ta + R_cod*Ta + R_hx*Tc)/(R_x+R_cod+R_hx);
        else
            Ts = (R_r*Ta + R_hr*Tc)/(R_r+R_hr);
        end


        if ( mod(N,step)==0 )
            output_T((N/step),n) = Ts-273.15;
            output_Q((N/step),n) = sum(Q(:));
        end

    end
end
%%
output_T_filter = output_T';
output_Q_filter = output_Q';
index = max(output_T_filter, [], 2)>= 310;
output_T_filter = output_T_filter(index, :);
output_Q_filter = output_Q_filter(index, :);
Sample_filter = Sample(index, :);
dataset = [output_T_filter, output_Q_filter, Sample_filter];

randomOrder = randperm(size(dataset, 1));
shuffledDataset = dataset(randomOrder, :);

%% Write dataset of input and features;
writematrix(shuffledDataset, 'dataset.csv');