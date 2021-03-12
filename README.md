# NaiveGo

This is a simple gomoku program, which is a reproduction version of the AlphaZero paper.

## Setting up and Running NaiveGo

If you just want to play with the provided model, run the following order in the current directory

    python gui.py

And if you want to train the neural network, run the following order in the current directory

    python train.py

## File structure

    gui.py: Interface for playing with the NavieGomoku

    train.py: Interface for training neural network

    gomoku_board.py: Realization of the gomoku board state

    mcts.py: Performing the Monte Carlo Tree Search algorithm

    alpha.py: Performing the AlphaZero Version MCTS algorithm

    policy_network.py: Neural network to evaluate the current board state.
