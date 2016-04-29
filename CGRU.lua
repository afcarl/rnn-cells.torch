require 'nngraph'

--[[
Implementation of convolutional GRU:
http://arxiv.org/abs/1511.08228

All equations are from page 3.
]]

function CGRU(stateSize)
   local s = nn.Identity()()

   local u = nn.Sigmoid()(
            nn.SpatialConvolutionMM(stateSize, stateSize, 3, 3, 1, 1, 1, 1)(s))

   local oneMinusU = nn.AddConstant(1, false)(nn.MulConstant(-1, false)(u))

   local r = nn.Sigmoid()(
            nn.SpatialConvolutionMM(stateSize, stateSize, 3, 3, 1, 1, 1, 1)(s))

   local rS = nn.CMulTable()({r, s})

   local newS = nn.Tanh()(
            nn.SpatialConvolutionMM(stateSize, stateSize, 3, 3, 1, 1, 1, 1)(rS))

   local uS = nn.CMulTable()({u, s})

   local oneMinusUNewS = nn.CMulTable()({oneMinusU, newS})

   local newS = nn.CAddTable()({uS, oneMinusUNewS})

   return nn.gModule({s}, {newS})
end
