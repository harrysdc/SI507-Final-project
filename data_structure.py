import pandas as pd 
import numpy as np 
from collections import Counter


class Node: 
    def __init__(self, Y: list, X: pd.DataFrame, min_samples_split=None, max_depth=None, depth=None, node_type=None, rule=None):
        self.Y = Y 
        self.X = X
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5
        self.depth = depth if depth else 0
        self.features = list(self.X.columns)
        self.node_type = node_type if node_type else 'root'
        self.rule = rule if rule else ""
        self.counts = Counter(Y)
        self.gini_impurity = self.get_GINI()

        # Sort counts and save the final prediction of a node 
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        self.yhat = yhat 
        self.n = len(Y)

        # init left/right nodes as empty
        self.left = None 
        self.right = None 
        self.best_feature = None 
        self.best_value = None 


    def GINI_impurity(y1_count, y2_count):
        """
        Given the observations of a binary class calculate the GINI impurity
        """
        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

        n = y1_count + y2_count
        if n == 0:
            return 0.0

        p1 = y1_count / n
        p2 = y2_count / n
        gini = 1 - (p1 ** 2 + p2 ** 2)

        return gini


    def ma(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI(self):
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)
        return self.GINI_impurity(y1_count, y2_count)

    def best_split(self):
        df = self.X.copy()
        df['Y'] = self.Y
        GINI_base = self.get_GINI()
        max_gain = 0

        best_feature = None
        best_value = None
        for feature in self.features:
            Xdf = df.dropna().sort_values(feature)
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                left_counts = Counter(Xdf[Xdf[feature]<value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature]>=value]['Y'])

                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                gini_left = self.GINI_impurity(y0_left, y1_left)
                gini_right = self.GINI_impurity(y0_right, y1_right)

                n_left = y0_left + y1_left
                n_right = y0_right + y1_right
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                wGINI = w_left * gini_left + w_right * gini_right
                GINIgain = GINI_base - wGINI
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 
                    max_gain = GINIgain

        return (best_feature, best_value)

    def build_tree(self):
        """
        grow tree recurisvely
        """
        df = self.X.copy()
        df['Y'] = self.Y

        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value

                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                left = Node(left_df['Y'].values.tolist(), left_df[self.features], depth=self.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split, node_type='left_node',rule=f"{best_feature} <= {round(best_value, 3)}")
                self.left = left 
                self.left.build_tree()

                right = Node(right_df['Y'].values.tolist(), right_df[self.features], depth=self.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split,node_type='right_node',rule=f"{best_feature} > {round(best_value, 3)}")
                self.right = right
                self.right.build_tree()


    def predict_obs(self, values):
        """
        predict the class given a set of features
        """
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return cur_node.yhat
        

    def predict(self, X:pd.DataFrame):
        """
        Batch prediction
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

