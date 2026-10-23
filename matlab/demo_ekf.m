% DEMO_EKF  Simulation d'une rotation de 45° en pitch avec biais gyroscope
%
%   Compatible MATLAB et GNU Octave :
%     >> cd matlab
%     >> demo_ekf

fs = 100;                 % Fréquence d'échantillonnage [Hz]
duration = 10;            % Durée [s]
g = 9.81;
sigmaA = 0.04;            % Écart-type bruit accéléromètre [m/s^2]
sigmaG = deg2rad(0.05);   % Écart-type bruit gyroscope [rad/s]
biasTrue = deg2rad([0.5; -0.3; 0.2]);

N = duration * fs;
t = (0:N-1)' / fs;
pitchTrue = deg2rad(45) * min(t / 5, 1);   % Rampe sur 5 s puis palier
rate = [diff(pitchTrue) * fs; 0];

rand('seed', 0); randn('seed', 0);
ekf = QuaternionEKF('SampleRate', fs);
rpy = zeros(N, 3);
bias = zeros(N, 3);

for k = 1:N
    th = pitchTrue(k);
    accel = g * [-sin(th); 0; cos(th)] + sigmaA * randn(3, 1);
    gyro = [0; rate(k); 0] + biasTrue + sigmaG * randn(3, 1);

    ekf.predict(gyro);
    ekf.update(accel);

    rpy(k, :) = ekf.eulerAngles();
    bias(k, :) = ekf.gyroBias()';
end

err = rad2deg(rpy(:, 2) - pitchTrue);
fprintf('RMSE pitch : %.4f deg\n', sqrt(mean(err.^2)));
fprintf('Biais estimé (x, y) : [%.3f %.3f] deg/s (vrai [%.3f %.3f])\n', ...
        rad2deg(bias(end, 1:2)), rad2deg(biasTrue(1:2)));

try
    figure;
    subplot(2, 1, 1);
    plot(t, rad2deg(pitchTrue), 'k--', t, rad2deg(rpy(:, 2)));
    xlabel('Temps [s]'); ylabel('Pitch [deg]'); legend('Vrai', 'EKF'); grid on;
    subplot(2, 1, 2);
    plot(t, rad2deg(bias(:, 1:2)));
    xlabel('Temps [s]'); ylabel('Biais [deg/s]'); legend('b_{gx}', 'b_{gy}'); grid on;
catch
    disp('Affichage graphique indisponible (mode sans écran).');
end
