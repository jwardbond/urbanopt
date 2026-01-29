\ Model pathway_optimizer
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  104406.91 x_1 + 103935.83 x_2 + 229522.1132079158 x_3
   + 101788.6101458127 x_4 + 64989.6 x_5 + 48832.28705854077 x_6
   + 37215.67 x_7 + 61353.35171219946 x_8 + 42526.28 x_9
   + 45827.39838827719 x_10
Subject To
 R0: 5 x_1 + 5 x_2 + 10 x_3 + 5 x_4 + 4 x_5 + 2 x_6 + 2 x_7 + 3 x_8 + 2 x_9
   + 2 x_10 >= 20
 R1: x_1 + x_5 <= 1
 R2: x_1 + x_6 <= 1
 R3: x_2 + x_7 <= 1
 R4: x_2 + x_5 <= 1
 R5: x_2 + x_6 <= 1
 R6: x_3 + x_5 <= 1
 R7: x_3 + x_6 <= 1
 R8: x_3 + x_8 <= 1
 R9: x_4 + x_7 <= 1
 R10: x_4 + x_5 <= 1
 R11: x_4 + x_6 <= 1
 R12: x_4 + x_10 <= 1
 R13: x_4 + x_8 <= 1
 R14: x_2 + x_5 <= 1
 R15: x_1 + x_5 <= 1
 R16: x_4 + x_5 <= 1
 R17: x_3 + x_5 <= 1
 R18: x_2 + x_6 <= 1
 R19: x_1 + x_6 <= 1
 R20: x_4 + x_6 <= 1
 R21: x_3 + x_6 <= 1
 R22: x_2 + x_7 <= 1
 R23: x_4 + x_7 <= 1
 R24: x_4 + x_8 <= 1
 R25: x_3 + x_8 <= 1
 R26: x_4 + x_10 <= 1
Bounds
Binaries
 x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_10
End
