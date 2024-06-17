class MinMaxValueConstraint(tf.keras.constraints.Constraint):
    """Constrain the weights between two values."""
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


class Tension_Jump_Activation(Layer):
    def __init__(self, min_T=0.0, max_T=1.0, min_K=0.0, max_K=1.0, min_B=0.5, max_B=8.5):
        super(Tension_Jump_Activation, self).__init__()
        self.T = self.add_weight(name='Threshold', shape=(),
                                 initializer=RandomUniform(minval=min_T, maxval=max_T),
                                 trainable=True,
                                 constraint=MinMaxValueConstraint(min_T, max_T))
        self.K = self.add_weight(name='Resistance', shape=(),
                                 initializer=RandomUniform(minval=min_K, maxval=max_K),
                                 trainable=True,
                                 constraint=MinMaxValueConstraint(min_K, max_K))
        self.B = self.add_weight(name='Baseline', shape=(),
                                 initializer=RandomUniform(minval=min_B, maxval=max_B),
                                 trainable=True,
                                 constraint=MinMaxValueConstraint(min_B, max_B))

    def call(self, inputs):
        # Linear response up to the threshold simulating stable tension
        stable_output = self.K * inputs
        # Exponential response beyond the threshold simulating a break in surface
        breaking_output = self.B * self.T + tf.exp(self.K * (inputs - self.T))

        # Switch behavior at the threshold
        return tf.where(inputs <= self.T, stable_output, breaking_output)
