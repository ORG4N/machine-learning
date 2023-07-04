function [net, tr] = neuralNetwork(x,t)
    
    net = patternnet(10, "trainscg", "mse");
    
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    [net, tr] = train(net, x, t);

end