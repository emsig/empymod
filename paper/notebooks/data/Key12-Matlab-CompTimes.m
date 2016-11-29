clear all; close all; clc

f   = 1;        % Frequency (Hz)
zTx = 990;      % Depth of transmitter (m)


for il = 1:2
    
    if il == 1  % 5 layers:
        disp('*** 5 layers')
        %       Air    Ocean    Seafloor--->
        z   = [-1d6    0        1000    2000    2100    ];   % Layer top depths (the first value is not used)
        rho = [1d12    0.3      1       100     1       ];   % Layer resistivities (ohm-m)
        nlay = 5; 
        alloff = [1,5,21,81,321];
    else  % 100 layers:
        disp('*** 100 layers')
        z   = [-1d6     0       1000    2000    2100+linspace(0,10000,96) ];   % Layer top depths
        rho = [1d12   .3        1       100     1+0*linspace(0,10000,96) ];    % Layer resistivities (ohm-m)
        nlay = 100;
        alloff = [21,81,321];
    end
    
    for i = 1:size(alloff, 2)
        noff = alloff(i);

        r   = linspace(500,20000,noff);  % Ranges (m) to receivers
        zRx = 1000*ones(size(r));        % Depth of receivers   (m)   

        disp(['*** offsets : ',num2str(noff),';  Layers : ',num2str(nlay)])

        sig = 1./rho;

        % QWE
        relTol = 1d-6;
        absTol = 1d-24;
        nQuad  = 9; 
        f_q9 = @() get_CSEM1D_FD_QWE(f,r,zRx,zTx,z,sig,relTol,absTol,nQuad);
        t_q9 = timeit(f_q9);
        str = sprintf('%10.0f ms :: 9pt QWE nospl', t_q9*1000);
        disp(str);

        % QWE spline
        lSpline = true;
        nPtsPerDecade = 40;

        relTol = 1d-2; 
        absTol = 1d-24;
        nQuad  = 9; 
        f_q9Spl = @() get_CSEM1D_FD_QWE(f,r,zRx,zTx,z,sig,relTol,absTol,nQuad,lSpline,nPtsPerDecade);
        t_q9Spl = timeit(f_q9Spl);
        str = sprintf('%10.0f ms :: 9pt splne', t_q9Spl*1000);
        disp(str);

        % 201pt filter
        fkk201 = @() get_CSEM1D_FD_FHT (f,r,zRx,zTx,z,sig,'kk201Hankel.txt');
        tkk201 = timeit(fkk201);
        str = sprintf('%10.0f ms :: 201pt nospl', tkk201*1000);
        disp(str);

        % 201pt filter lagged
        lUseLaggedConv = true;
        fkk201_lag = @() get_CSEM1D_FD_FHT(f,r,zRx,zTx,z,sig,'kk201Hankel.txt',lUseLaggedConv);
        tkk201_lag = timeit(fkk201_lag);
        str = sprintf('%10.0f ms :: 201pt splne', tkk201_lag*1000);
        disp(str);

        % 801pt filter
        fwa801 = @() get_CSEM1D_FD_FHT(f,r,zRx,zTx,z,sig,'wa801Hankel.txt');
        twa801 = timeit(fwa801);
        str = sprintf('%10.0f ms :: 801pt nospl', twa801*1000);
        disp(str);

        % 801pt filter lagged
        lUseLaggedConv = true;
        fwa801_lag = @() get_CSEM1D_FD_FHT(f,r,zRx,zTx,z,sig,'wa801Hankel.txt',lUseLaggedConv);
        twa801_lag = timeit(fwa801_lag);
        str = sprintf('%10.0f ms :: 801pt splne', twa801_lag*1000);
        disp(str);
    end
end
