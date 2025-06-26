graph [
  directed 1
  comment "Automatically generated from task scheduling data"
  node [
    id 0
    label "0"
    OwnerJobID "999"
    CPUNeed_Claimed "543013"
    RAMNeed_Claimed "3520"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "12"
    TimeStartOnMachine "[5"
    TimeFinishOnMachine "6]"
  ]
  node [
    id 1
    label "1"
    OwnerJobID "999"
    CPUNeed_Claimed "586599"
    RAMNeed_Claimed "2657"
    LastStatus "Rejected"
    PriorityNo "0"
    TimeStartOnMachine "[0]"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 7
    label "2"
    OwnerJobID "999"
    CPUNeed_Claimed "595563"
    RAMNeed_Claimed "953"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "12"
    TimeStartOnMachine "[6]"
    TimeFinishOnMachine "[1]"
  ]
  node [
    id 8
    label "3"
    OwnerJobID "999"
    CPUNeed_Claimed "551977"
    RAMNeed_Claimed "182"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "12"
    TimeStartOnMachine "[0]"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 9
    label "4"
    OwnerJobID "999"
    CPUNeed_Claimed "568385"
    RAMNeed_Claimed "3808"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "12"
    TimeStartOnMachine "[2]"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 10
    label "5"
    OwnerJobID "999"
    CPUNeed_Claimed "616696"
    RAMNeed_Claimed "1797"
    LastStatus "Rejected"
    PriorityNo "0"
    TimeStartOnMachine "3]"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 11
    label "6"
    OwnerJobID "999"
    CPUNeed_Claimed "556101"
    RAMNeed_Claimed "2991"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "12"
    TimeStartOnMachine "[1"
    TimeFinishOnMachine "4"
  ]
  node [
    id 12
    label "7"
    OwnerJobID "997"
    CPUNeed_Claimed "617851"
    RAMNeed_Claimed "2528"
    LastStatus "Rejected"
    PriorityNo "0"
    TimeStartOnMachine "-1"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 13
    label "8"
    OwnerJobID "997"
    CPUNeed_Claimed "559338"
    RAMNeed_Claimed "2583"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "38"
    TimeStartOnMachine "[1"
    TimeFinishOnMachine "5"
  ]
  node [
    id 14
    label "9"
    OwnerJobID "997"
    CPUNeed_Claimed "591640"
    RAMNeed_Claimed "385"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "38"
    TimeStartOnMachine "[6]"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 2
    label "10"
    OwnerJobID "997"
    CPUNeed_Claimed "616009"
    RAMNeed_Claimed "3556"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "38"
    TimeStartOnMachine "[8"
    TimeFinishOnMachine "9]"
  ]
  node [
    id 3
    label "11"
    OwnerJobID "991"
    CPUNeed_Claimed "565049"
    RAMNeed_Claimed "2424"
    LastStatus "N_0_of_Clst_0_of_DC_0_of_CP_Clouderino61"
    PriorityNo "31"
    TimeStartOnMachine "[1"
    TimeFinishOnMachine "2"
  ]
  node [
    id 4
    label "12"
    OwnerJobID "989"
    CPUNeed_Claimed "599029"
    RAMNeed_Claimed "805"
    LastStatus "N_0_of_FP_0_of_Cyphocom76"
    PriorityNo "33"
    TimeStartOnMachine "[8"
    TimeFinishOnMachine "10"
  ]
  node [
    id 5
    label "13"
    OwnerJobID "989"
    CPUNeed_Claimed "612263"
    RAMNeed_Claimed "327"
    LastStatus "Rejected"
    PriorityNo "0"
    TimeStartOnMachine "12]"
    TimeFinishOnMachine "-1"
  ]
  node [
    id 6
    label "14"
    OwnerJobID "988"
    CPUNeed_Claimed "611771"
    RAMNeed_Claimed "3541"
    LastStatus "Rejected"
    PriorityNo "0"
    TimeStartOnMachine "11"
    TimeFinishOnMachine "12"
  ]
  edge [
    source 13
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 1
    target 6
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 7
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 11
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 10
    target 11
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 8
    dependency_type "non-immediate"
  ]
  edge [
    source 8
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 11
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 11
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 3
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 9
    dependency_type "non-immediate"
  ]
  edge [
    source 12
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 10
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 5
    dependency_type "non-immediate"
  ]
  edge [
    source 2
    target 3
    dependency_type "non-immediate"
  ]
  edge [
    source 10
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 8
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 14
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 8
    target 9
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 9
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 8
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 4
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 11
    dependency_type "non-immediate"
  ]
  edge [
    source 12
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 13
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 10
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 3
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 7
    target 3
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 8
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 5
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 11
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 10
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 6
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 3
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 9
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 4
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 7
    target 11
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 10
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 12
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 6
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 10
    dependency_type "non-immediate"
  ]
  edge [
    source 11
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 1
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 10
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 0
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 6
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 8
    target 10
    dependency_type "non-immediate"
  ]
  edge [
    source 5
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 13
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 0
    target 7
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 4
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 5
    dependency_type "non-immediate"
  ]
  edge [
    source 8
    target 11
    dependency_type "non-immediate"
  ]
  edge [
    source 0
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 2
    target 4
    dependency_type "non-immediate"
  ]
  edge [
    source 8
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 14
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 10
    dependency_type "non-immediate"
  ]
  edge [
    source 13
    target 3
    dependency_type "non-immediate"
  ]
  edge [
    source 8
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 2
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 7
    target 13
    dependency_type "non-immediate"
  ]
  edge [
    source 3
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 1
    target 11
    dependency_type "non-immediate"
  ]
  edge [
    source 8
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 11
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 10
    target 2
    dependency_type "non-immediate"
  ]
  edge [
    source 9
    target 1
    dependency_type "immediate"
  ]
  edge [
    source 0
    target 1
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 12
    dependency_type "non-immediate"
  ]
  edge [
    source 12
    target 14
    dependency_type "non-immediate"
  ]
  edge [
    source 7
    target 4
    dependency_type "non-immediate"
  ]
  edge [
    source 1
    target 13
    dependency_type "non-immediate"
  ]
]
