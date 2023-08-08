classdef Link
    %LINK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        isPrismatic = false;

        alpha;
        a;
        d;
        theta;

        m;
        i_r_ci;
        I;
    end
    
    methods
        function obj = Link(alpha,a,d,theta)
            %LINK Construct an instance of this class
            %   Detailed explanation goes here
            obj.alpha = alpha;
            obj.a = a;
            obj.d = d;
            obj.theta = theta;
        end

        function obj = setInertia(Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
            obj.I = [Ixx, Ixy, Ixz; Ixy Iyy, Iyz; Ixz, Iyz Izz];
        end
    end
end

