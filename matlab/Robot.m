classdef Robot
    %ROBOT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        DH_table = []
        links = []
    end
    
    methods
        function obj = Robot()
            %ROBOT Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function obj = addLink(obj,link)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            if ~isa(link, 'Link')
                error('Expected a link as input');
            end
            obj.DH_table = [obj.DH_table;link.alpha link.a link.d link.theta];
            obj.links = [obj.links, link];
        end

        function computeInertia(obj)
            [n, ~] = size(obj.DH_table);
            zi = [0;0;1];
            w0 = 0;
            v0 = 0;
            
            T = 0;

            for i=1:n
                H = DH_single_transform(obj.DH_table, i);
                rot_i = H(1:3, 1:3);
                r_i = H(1:3, 4);

                currentLink = obj.links(i);

                
            end

        end

        function computeC()

        end

        function computeGravity(g)

        end

        function getDynamicModel()
            
        end
    end

end

