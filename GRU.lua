require 'nngraph'

--[[
Implementation of GRU from:
http://arxiv.org/pdf/1412.3555v1.pdf
]]

function GRU(inputSize, stateSize)
   local input = nn.Identity()()
   local prevH = nn.Identity()()
      
   local inputPrevH = nn.JoinTable(2)({input, prevH})
   local concatSize = inputSize + stateSize

   -- update gate (page 4)
   local z = nn.Sigmoid()(nn.Linear(concatSize, stateSize)(inputPrevH))
   local oneMunisZ = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z))

   -- reset gate (page 4)
   local r = nn.Sigmoid()(nn.Linear(concatSize, stateSize)(inputPrevH))
   
   -- candidate activations
   local resetHidden = nn.CMulTable()({r, prevH})
   local inputResetHidden = nn.JoinTable(2)({input, resetHidden})
   local hHat = nn.Tanh()(nn.Linear(concatSize, stateSize)(inputResetHidden))
    
    -- compute new interpolated hidden state, based on the update gate
    local zHHat = nn.CMulTable()({z, hHat})
    local oneMunisZPrevH = nn.CMulTable()({oneMunisZ, prevH})
    
    local nextH = nn.CAddTable()({zHHat, oneMunisZPrevH})
    
    return nn.gModule({input, prevH}, {nextH})
end
