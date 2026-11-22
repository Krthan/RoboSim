This is a simple URDFParser which works for serial manipulators when an URDF file is provided, examples of usage given in `GetRobot.py`. 

- RNEA algorithm for Forward dynamics
- CRBA and ABA algorithm for Inverse dynamics 
- Levenberg-Marquardt Algorithm and Non-linear optimization problem for Inverse Kinematics

This simulator uses screw and twist representations of the robot body for calculatiosns

ToDo:
1. ~~Implement ABA algorithm~~
2. ~~Implement Inverse Kinematics~~
3. Make it moduluar to produce symbolic equations
4. Improve CRBA and solving with sparsity-based solving
