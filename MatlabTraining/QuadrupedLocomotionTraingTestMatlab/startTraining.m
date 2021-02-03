%% OPEN THE QUADRUPED ROBOT MODEL
initializeRobotParameters;
mdl = 'rlQuadrupedRobot';
open_system(mdl);

%% CREATE ENVIRONMENT INTERFACE
% Specify the parameters for the observation set
numObs = 44;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = 'observations';

% Specify the parameters for the action set
numAct = 8;
actInfo = rlNumericSpec([numAct 1],'LowerLimit',-1,'UpperLimit', 1);
actInfo.Name = 'torque';

% Create the environment using the reinforcement learning model
blk = [mdl, '/RL Agent'];
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);

% The reset function introduces random deviations into the initial joint angles and angular velocities during training
env.ResetFcn = @quadrupedResetFcn;

%% Create DDPG agent
% Create the networks in the MATLAB workspace
createNetworks;

% View the critic network configuration
plot(criticNetwork);

% Specify the agent options using rlDDPGAgentOptions
agentOptions = rlDDPGAgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 250;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.NoiseOptions.MeanAttractionConstant = 0.15;
agentOptions.NoiseOptions.Variance = 0.1;

% Create the rlDDPGAgent object for the agent
agent = rlDDPGAgent(actor,critic,agentOptions);

%% Specify Training Options
maxEpisodes = 10000;
maxSteps = floor(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxEpisodes,...
    'MaxStepsPerEpisode',maxSteps,...
    'ScoreAveragingWindowLength',250,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',190,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',200);

% train the agent in parallel training mode
trainOpts.UseParallel = false;
trainOpts.ParallelizationOptions.Mode = 'async';
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

%% Train Agent
doTraining = true;
if doTraining
    % Train the agent
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent for the example
    load('rlQuadrupedAgent.mat','agent')
end