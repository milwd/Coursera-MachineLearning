function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; 

% scatter(x, y)
% xlabel('population');
% ylabel('revenue');

plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s'); 
xlabel('Population of City in 10,000s');

end