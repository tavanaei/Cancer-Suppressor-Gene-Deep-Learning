local M = {}

function M.Parse(arg)	
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Cancer Specification Program')
	cmd:text('Options:')
	cmd:text()

	cmd:option('-positive','OG_Map','Positive directory')
	cmd:option('-negative','SG_Map','Negative Directory')
	cmd:option('-neutral','UR_Map','Neutral Directory')
	cmd:option('-GPU',1,'preferred GPU')
	cmd:option('-nGPU',1,'No of GPUs')
	cmd:option('-kernel','16,32,32,64,64','Kernels for convolution layers')
	cmd:option('-stride','4,2,2,2','Stride values for Pooling')
	cmd:option('-hidden','100,50','Hidden Layers')
	cmd:option('-iterations',1,'No of iterations')
	cmd:option('-batchSize',10,'Batch size')
	cmd:option('-learningRate',0.01,'Learning rate')
	cmd:option('-learningRateDecay',0.00001,'Learning rate decay')
	cmd:option('-momentum',0.6,'Weight change history')
	cmd:option('-weightDecay',1e-4,'regularizer parameter')
	cmd:option('-cuda',false,'Use Cuda')
	cmd:option('-p',5,'Kernel Size')
	cmd:option('-trainSize',2000,'Training Samples')
	cmd:option('-testSize',350,'Testing Samples')
	cmd:option('-validSize',10,'Validation Samples')
	cmd:option('-model','Model.t7','Model File')
	cmd:option('-result','ResTest.dat','Test Results of Target vs Predict')
cmd:text()

	local opt=cmd:parse(arg or {})
	local ks = {}
	local ss = {}
	local hs = {}
	local index = 1
	for i,v in opt.kernel:gmatch("%d+") do
		ks[index] = i
		index = index+1
	end
	index=1
	for i,v in opt.stride:gmatch("%d+") do
    ss[index] = i
		index = index+1
  end
	index=1
	for i,v in opt.hidden:gmatch("%d+") do
    hs[index] = i
		index = index+1
  end

	opt.kernel = ks
	opt.stride = ss
	opt.hidden = hs

	return opt
end

return M
