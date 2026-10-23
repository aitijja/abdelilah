classdef QuaternionEKF < handle
    % QUATERNIONEKF  Filtre de Kalman étendu pour l'orientation (MPU6050)
    %
    %   État : x = [q0 q1 q2 q3 bgx bgy bgz]'
    %     q   : quaternion d'orientation (capteur -> monde), scalaire en tête
    %     b_g : biais du gyroscope [rad/s]
    %
    %   Portage MATLAB/Octave de src/fusion/kalman_filter.py, mêmes
    %   conventions et mêmes équations (voir docs/SENSOR_FUSION.md, §1).
    %
    %   Exemple :
    %     ekf = QuaternionEKF('SampleRate', 100);
    %     for k = 1:N
    %         ekf.predict(gyro(k,:)');   % rad/s
    %         ekf.update(accel(k,:)');   % m/s^2
    %         rpy(k,:) = ekf.eulerAngles();
    %     end

    properties
        x           % Vecteur d'état (7x1)
        P           % Covariance de l'état (7x7)
        Q           % Bruit de processus (7x7)
        R           % Bruit de mesure accéléromètre (3x3)
        dt          % Période d'échantillonnage [s]
        g           % Gravité [m/s^2]
        accelThresholdLow   % Seuil bas de détection de mouvement [m/s^2]
        accelThresholdHigh  % Seuil haut de détection de mouvement [m/s^2]
    end

    methods
        function obj = QuaternionEKF(varargin)
            % QUATERNIONEKF  Constructeur, paramètres nom/valeur :
            %   'InitialOrientation'  quaternion initial (défaut [1 0 0 0])
            %   'GyroNoiseDensity'    rad/s/sqrt(Hz)    (défaut 8.73e-5)
            %   'GyroBiasRandomWalk'  rad/s^2/sqrt(Hz)  (défaut 8.73e-8)
            %   'AccelNoiseDensity'   m/s^2/sqrt(Hz)    (défaut 3.924e-3)
            %   'SampleRate'          Hz                (défaut 100)
            %   'Gravity'             m/s^2             (défaut 9.81)
            p = inputParser;
            p.addParameter('InitialOrientation', [1 0 0 0]);
            p.addParameter('GyroNoiseDensity', 8.73e-5);
            p.addParameter('GyroBiasRandomWalk', 8.73e-8);
            p.addParameter('AccelNoiseDensity', 3.924e-3);
            p.addParameter('SampleRate', 100);
            p.addParameter('Gravity', 9.81);
            p.parse(varargin{:});
            o = p.Results;

            obj.dt = 1 / o.SampleRate;
            obj.g = o.Gravity;

            q0 = o.InitialOrientation(:);
            obj.x = [q0 / norm(q0); zeros(3, 1)];

            obj.P = blkdiag(0.1 * eye(4), 0.01 * eye(3));

            obj.Q = blkdiag((o.GyroNoiseDensity^2) * obj.dt * eye(4), ...
                            (o.GyroBiasRandomWalk^2) * obj.dt * eye(3));
            obj.R = (o.AccelNoiseDensity^2) * o.SampleRate * eye(3);

            obj.accelThresholdLow = 0.8 * obj.g;
            obj.accelThresholdHigh = 1.2 * obj.g;
        end

        function q = quaternion(obj)
            % Quaternion d'orientation estimé (4x1)
            q = obj.x(1:4);
        end

        function b = gyroBias(obj)
            % Biais du gyroscope estimé [rad/s] (3x1)
            b = obj.x(5:7);
        end

        function rpy = eulerAngles(obj)
            % Angles d'Euler ZYX [roll pitch yaw] en radians
            q = obj.x(1:4);
            roll  = atan2(2 * (q(1)*q(2) + q(3)*q(4)), 1 - 2 * (q(2)^2 + q(3)^2));
            pitch = asin(max(-1, min(1, 2 * (q(1)*q(3) - q(4)*q(2)))));
            yaw   = atan2(2 * (q(1)*q(4) + q(2)*q(3)), 1 - 2 * (q(3)^2 + q(4)^2));
            rpy = [roll pitch yaw];
        end

        function predict(obj, gyro)
            % PREDICT  Propagation avec la mesure du gyroscope [rad/s]
            omega = gyro(:) - obj.x(5:7);

            % Intégration RK4 du quaternion puis normalisation
            qNew = QuaternionEKF.integrateRK4(obj.x(1:4), omega, obj.dt);
            obj.x(1:4) = qNew / norm(qNew);

            % Jacobienne F et matrice de transition Phi = I + F*dt
            F = zeros(7);
            F(1:4, 1:4) = 0.5 * QuaternionEKF.omegaMatrix(omega);
            F(1:4, 5:7) = -0.5 * QuaternionEKF.xiMatrix(obj.x(1:4));
            Phi = eye(7) + F * obj.dt;

            obj.P = Phi * obj.P * Phi' + obj.Q;
        end

        function used = update(obj, accel)
            % UPDATE  Correction avec l'accéléromètre [m/s^2]
            %   Retourne false si la mesure est rejetée (mouvement dynamique)
            accel = accel(:);
            aNorm = norm(accel);
            used = aNorm >= obj.accelThresholdLow && aNorm <= obj.accelThresholdHigh;
            if ~used
                return;
            end

            q = obj.x(1:4);
            y = accel - obj.predictGravity(q);       % Innovation
            H = obj.measurementJacobian(q);
            S = H * obj.P * H' + obj.R;
            K = (obj.P * H') / S;                    % Gain de Kalman

            obj.x = obj.x + K * y;
            obj.x(1:4) = obj.x(1:4) / norm(obj.x(1:4));

            % Forme de Joseph (stabilité numérique)
            IKH = eye(7) - K * H;
            obj.P = IKH * obj.P * IKH' + K * obj.R * K';
        end

        function h = predictGravity(obj, q)
            % Gravité prédite dans le repère capteur : R(q)' * [0 0 g]'
            h = obj.g * [2 * (q(2)*q(4) - q(1)*q(3));
                         2 * (q(1)*q(2) + q(3)*q(4));
                         q(1)^2 - q(2)^2 - q(3)^2 + q(4)^2];
        end

        function H = measurementJacobian(obj, q)
            % Jacobienne H = dh/dx (3x7), dh/db_g = 0
            H = zeros(3, 7);
            H(:, 1:4) = obj.g * [-2*q(3)  2*q(4) -2*q(1)  2*q(2);
                                  2*q(2)  2*q(1)  2*q(4)  2*q(3);
                                  2*q(1) -2*q(2) -2*q(3)  2*q(4)];
        end

        function Rm = rotationMatrix(obj)
            % Matrice de rotation capteur -> monde (3x3)
            q = obj.x(1:4);
            Rm = [1 - 2*(q(3)^2 + q(4)^2), 2*(q(2)*q(3) - q(1)*q(4)), 2*(q(2)*q(4) + q(1)*q(3));
                  2*(q(2)*q(3) + q(1)*q(4)), 1 - 2*(q(2)^2 + q(4)^2), 2*(q(3)*q(4) - q(1)*q(2));
                  2*(q(2)*q(4) - q(1)*q(3)), 2*(q(3)*q(4) + q(1)*q(2)), 1 - 2*(q(2)^2 + q(3)^2)];
        end
    end

    methods (Static)
        function W = omegaMatrix(w)
            % Matrice Omega(w) telle que dq/dt = 0.5 * Omega(w) * q
            W = [0    -w(1) -w(2) -w(3);
                 w(1)  0     w(3) -w(2);
                 w(2) -w(3)  0     w(1);
                 w(3)  w(2) -w(1)  0];
        end

        function X = xiMatrix(q)
            % Matrice Xi(q) (4x3) : dérivée de dq/dt par rapport à omega
            X = [-q(2) -q(3) -q(4);
                  q(1) -q(4)  q(3);
                  q(4)  q(1) -q(2);
                 -q(3)  q(2)  q(1)];
        end

        function qNew = integrateRK4(q, w, dt)
            % Intégration Runge-Kutta 4 de dq/dt = 0.5 * Omega(w) * q
            W = 0.5 * QuaternionEKF.omegaMatrix(w);
            k1 = W * q;
            k2 = W * (q + 0.5 * dt * k1);
            k3 = W * (q + 0.5 * dt * k2);
            k4 = W * (q + dt * k3);
            qNew = q + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
        end
    end
end
