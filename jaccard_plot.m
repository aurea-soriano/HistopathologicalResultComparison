function jaccard_plot()
  X = 1:20;
  %Y = [6.34 4.22 12.88 8.80 2.73 7.32 5.45 4.98 12.97 2.78 4.7 6.43 7.82 4.88 3.56 4.2 6.1 3.99 5.77 4.59];
  Y = [0.9180284586 0.8911484148 0.922026329 0.8693145669 0.9117346208 0.9147925 0.8350516793 0.9064881835 0.9013723025 0.9030480575 0.8548498367 0.8660877994 0.9319634465 0.92221873 0.8662117669 0.8708135649 0.8664482591 0.9414502467 0.9203662826 0.8862011881]
  %mean(Y);
  %plot(X,Y,'.','MarkerSize',30) %'LineWidth',3,
  h = bar(Y);
  xlim([0 21])
  set (h(1), "basevalue", 0.8); 
  %set(gca,'YTick',(0:2.0:14),'XTick',(1:1:20), 'FontSize', 15) % 'YTick', (0:0.01:1)
  xlabel('User', 'FontSize', 20, 'FontName', 'times')
  ylabel('Jaccard', 'FontSize', 20, 'FontName', 'times'); % jaccard
  %ylabel('Jaccard', 'FontSize', 20, 'FontName', 'times'); 
  grid on
end
