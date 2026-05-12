''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *

_EPS = 1e-8   # numerical guard inside sqrt and sigmoid operations


# ---------------------------------------------------------------------------
# Physics / topology-inspired loss configuration
# ---------------------------------------------------------------------------

class PhysicsLossConfig:
    """Configuration for optional physics/topology-inspired auxiliary losses.

    All auxiliary losses are *disabled by default*.  Pass an instance to
    SR_NETWORK (and PhIREGANs.pretrain) to enable them.

    When use_aux_losses=False (default) the SR_NETWORK graph is built
    identically to the original baseline — no extra TF ops are created.

    Parameters
    ----------
    use_aux_losses : bool
        Master switch.  False → baseline behaviour preserved exactly.
    mu : list[float] | None
        Per-channel normalisation mean, e.g. [0.7684, -0.4575] for [u, v].
        Required when use_aux_losses=True.
    sig : list[float] | None
        Per-channel normalisation std, e.g. [5.02455, 5.9017] for [u, v].
        Required when use_aux_losses=True.
    lambda_speed : float
        Weight for L_speed (physical speed MSE).  Default 0.
    lambda_grad : float
        Weight for L_grad (speed gradient-magnitude MSE).  Default 0.
    lambda_wpd : float
        Weight for L_wpd (wind-power-density proxy MAE on speed^3).  Default 0.
    lambda_levelset : float
        Weight for L_levelset (soft level-set / exceedance loss).  Default 0.
    levelset_temperature : float
        Sigmoid steepness k used as sigmoid(k*(speed - tau)).  Default 10.0.
    levelset_thresholds : list[float]
        Physical speed thresholds in m/s for the soft level-set loss.
        Default [5.0, 10.0, 15.0].
    lambda_crit : float
        Weight for L_crit (critical-value / topological-extrema proxy loss).
        Focuses MSE on GT local-speed maxima above an adaptive per-sample
        threshold.  Default 0 (disabled).
    crit_high_z : float
        Adaptive threshold for maxima: mean + crit_high_z * std.  Default 1.0.
    crit_include_minima : bool
        If True, also penalise error at GT local minima.  Default False.
    crit_low_z : float
        Adaptive threshold offset for minima (typically negative).  Default -1.0.
    crit_pool : int
        Neighbourhood radius for the max-pool local-maximum detector.  Default 3.
    diagnostic_mode : bool
        When True, individual loss tensors are stored in model.loss_breakdown
        so they can be fetched via sess.run without running a training step.
    """

    _DEFAULT_THRESHOLDS = [5.0, 10.0, 15.0]

    def __init__(
        self,
        use_aux_losses=False,
        mu=None,
        sig=None,
        lambda_speed=0.0,
        lambda_grad=0.0,
        lambda_wpd=0.0,
        lambda_levelset=0.0,
        levelset_temperature=10.0,
        levelset_thresholds=None,
        lambda_crit=0.0,
        crit_high_z=1.0,
        crit_include_minima=False,
        crit_low_z=-1.0,
        crit_pool=3,
        diagnostic_mode=False,
    ):
        self.use_aux_losses       = use_aux_losses
        self.mu                   = list(mu)  if mu  is not None else None
        self.sig                  = list(sig) if sig is not None else None
        self.lambda_speed         = float(lambda_speed)
        self.lambda_grad          = float(lambda_grad)
        self.lambda_wpd           = float(lambda_wpd)
        self.lambda_levelset      = float(lambda_levelset)
        self.levelset_temperature = float(levelset_temperature)
        self.levelset_thresholds  = (list(levelset_thresholds)
                                     if levelset_thresholds is not None
                                     else list(self._DEFAULT_THRESHOLDS))
        self.lambda_crit          = float(lambda_crit)
        self.crit_high_z          = float(crit_high_z)
        self.crit_include_minima  = bool(crit_include_minima)
        self.crit_low_z           = float(crit_low_z)
        self.crit_pool            = int(crit_pool)
        self.diagnostic_mode      = diagnostic_mode

        if use_aux_losses and (mu is None or sig is None):
            raise ValueError(
                'PhysicsLossConfig: mu and sig must be provided when '
                'use_aux_losses=True.'
            )


# ---------------------------------------------------------------------------
# Physical speed and gradient helpers (pure TF, no numpy)
# ---------------------------------------------------------------------------

def _denorm_speed(x_norm, mu_tf, sig_tf):
    """Denormalise per-channel [u, v] and return scalar wind speed.

    Parameters
    ----------
    x_norm : tf.Tensor, shape [batch, H, W, C=2]
        Normalised [u, v] tensor from the training graph.
    mu_tf : tf.Tensor, shape [C]
        Per-channel means as a TF constant.
    sig_tf : tf.Tensor, shape [C]
        Per-channel standard deviations as a TF constant.

    Returns
    -------
    speed : tf.Tensor, shape [batch, H, W]
        Physical wind speed sqrt(u_phys^2 + v_phys^2 + eps).
    """
    x_phys = sig_tf * x_norm + mu_tf          # [batch, H, W, C] broadcast
    u_phys = x_phys[..., 0]                   # [batch, H, W]
    v_phys = x_phys[..., 1]                   # [batch, H, W]
    return tf.sqrt(u_phys ** 2 + v_phys ** 2 + _EPS)


def _gradmag(speed):
    """Finite-difference gradient magnitude of a 2-D speed field.

    Uses backward differences on a [batch, H-1, W-1] interior grid so that
    both dx and dy share the same spatial support.

    Parameters
    ----------
    speed : tf.Tensor, shape [batch, H, W]

    Returns
    -------
    gm : tf.Tensor, shape [batch, H-1, W-1]
        sqrt(dx^2 + dy^2 + eps).
    """
    dx = speed[:, :-1, 1:] - speed[:, :-1, :-1]   # [batch, H-1, W-1]
    dy = speed[:, 1:, :-1] - speed[:, :-1, :-1]   # [batch, H-1, W-1]
    return tf.sqrt(dx ** 2 + dy ** 2 + _EPS)


def _critical_value_loss(speed_hr, speed_sr, high_z=1.0,
                         include_minima=False, low_z=-1.0, pool=3):
    """Critical-value / topological-extrema proxy loss.

    Focuses squared error at GT local-speed maxima that exceed an adaptive
    per-sample threshold, inspired by the critical-value loss of Kissi et al.
    for topology-aware scalar-field interpolation.  For superlevel-set topology
    high-speed local maxima correspond to births of connected components, so
    matching scalar values at those points helps preserve feature prominence and
    spatially anchored topological structure.

    Parameters
    ----------
    speed_hr, speed_sr : tf.Tensor, shape [B, H, W]
        Physical scalar wind speed for GT and SR respectively.
    high_z : float
        Adaptive threshold = mean + high_z * std.  Only maxima above this
        threshold are kept to suppress noisy low-amplitude local maxima.
    include_minima : bool
        If True, also penalise squared error at GT local minima below
        mean + low_z * std.
    low_z : float
        Adaptive threshold offset for the minima branch (typically negative).
    pool : int
        Neighbourhood size for the max-pool local-maximum detector.

    Returns
    -------
    L_crit : tf.Tensor, scalar float32
    """
    ksize   = [1, pool, pool, 1]
    strides = [1, 1, 1, 1]
    eps_t   = tf.constant(_EPS, dtype=tf.float32)

    s_hr_4d = speed_hr[..., tf.newaxis]            # [B, H, W, 1]

    # Local maximum: a pixel whose value equals its neighbourhood max.
    local_max = tf.nn.max_pool(s_hr_4d, ksize=ksize, strides=strides, padding='SAME')
    local_max = local_max[..., 0]                   # [B, H, W]
    is_local_max = speed_hr >= local_max - eps_t

    # Per-sample adaptive threshold over spatial axes.
    mean_hr      = tf.reduce_mean(speed_hr, axis=[1, 2], keepdims=True)  # [B,1,1]
    std_hr       = tf.math.reduce_std(speed_hr, axis=[1, 2], keepdims=True)
    threshold_hi = mean_hr + high_z * std_hr

    max_mask  = tf.cast(is_local_max & (speed_hr >= threshold_hi), tf.float32)
    sq_err    = tf.square(speed_sr - speed_hr)      # [B, H, W]
    mask_sum  = tf.maximum(tf.reduce_sum(max_mask), 1.0)
    L_max     = tf.reduce_sum(max_mask * sq_err) / mask_sum

    if not include_minima:
        return L_max

    # Optional minima branch: max-pool on negated speed finds local minima.
    local_min_neg = tf.nn.max_pool(-s_hr_4d, ksize=ksize, strides=strides, padding='SAME')
    local_min     = -local_min_neg[..., 0]          # [B, H, W]
    is_local_min  = speed_hr <= local_min + eps_t
    threshold_lo  = mean_hr + low_z * std_hr
    min_mask      = tf.cast(is_local_min & (speed_hr <= threshold_lo), tf.float32)
    min_mask_sum  = tf.maximum(tf.reduce_sum(min_mask), 1.0)
    L_min         = tf.reduce_sum(min_mask * sq_err) / min_mask_sum

    return L_max + L_min


# ---------------------------------------------------------------------------
# SR network
# ---------------------------------------------------------------------------

class SR_NETWORK(object):
    def __init__(self, x_LR=None, x_HR=None, r=None, status='pretraining',
                 alpha_advers=0.001, phys_cfg=None):

        status = status.lower()
        if status not in ['pretraining', 'training', 'testing']:
            print('Error in network status.')
            exit()

        self.x_LR, self.x_HR = x_LR, x_HR
        # Populated in diagnostic_mode; None otherwise.
        self.loss_breakdown = None

        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        if status in ['pretraining', 'training']:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=True)
        else:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=False)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'pretraining':
            # Base MSE loss (unchanged from original).
            base_loss = self.compute_losses(
                self.x_HR, self.x_SR, None, None, alpha_advers, isGAN=False
            )

            if phys_cfg is not None and phys_cfg.use_aux_losses:
                aux_total, breakdown = self._build_aux_losses(
                    self.x_HR, self.x_SR, phys_cfg
                )
                self.g_loss = base_loss + aux_total

                if phys_cfg.diagnostic_mode:
                    # Weighted contributions for logging.
                    self.loss_breakdown = {
                        'L_uv':         base_loss,
                        'L_speed':      breakdown['L_speed'],
                        'L_grad':       breakdown['L_grad'],
                        'L_wpd':        breakdown['L_wpd'],
                        'L_levelset':   breakdown['L_levelset'],
                        'L_crit':       breakdown['L_crit'],
                        'w_L_speed':    phys_cfg.lambda_speed    * breakdown['L_speed'],
                        'w_L_grad':     phys_cfg.lambda_grad     * breakdown['L_grad'],
                        'w_L_wpd':      phys_cfg.lambda_wpd      * breakdown['L_wpd'],
                        'w_L_levelset': phys_cfg.lambda_levelset * breakdown['L_levelset'],
                        'w_L_crit':     phys_cfg.lambda_crit     * breakdown['L_crit'],
                        'total_loss':   self.g_loss,
                    }
            else:
                # Baseline: g_loss is exactly the original MSE scalar.
                self.g_loss = base_loss

            self.d_loss      = None
            self.disc_HR     = None
            self.disc_SR     = None
            self.d_variables = None
            self.advers_perf = None
            self.content_loss   = None
            self.g_advers_loss  = None

        elif status == 'training':
            self.disc_HR = self.discriminator(self.x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            # GAN adversarial path: aux losses not applied here.
            loss_out = self.compute_losses(
                self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, alpha_advers, isGAN=True
            )
            self.g_loss        = loss_out[0]
            self.d_loss        = loss_out[1]
            self.advers_perf   = loss_out[2]
            self.content_loss  = loss_out[3]
            self.g_advers_loss = loss_out[4]

        else:
            self.g_loss      = None
            self.d_loss      = None
            self.disc_HR     = None
            self.disc_SR     = None
            self.d_variables = None
            self.advers_perf = None
            self.content_loss   = None
            self.g_advers_loss  = None


    def _build_aux_losses(self, x_HR, x_SR, cfg):
        """Build auxiliary physics/topology loss terms as TF ops.

        Only called when cfg.use_aux_losses=True.  All five loss tensors are
        always constructed so they can be fetched in diagnostic_mode even when
        their lambda is 0.

        Parameters
        ----------
        x_HR : tf.Tensor, shape [batch, H, W, C]
        x_SR : tf.Tensor, shape [batch, H, W, C]
        cfg  : PhysicsLossConfig

        Returns
        -------
        weighted_total : tf.Tensor, scalar
            Sum of lambda-weighted auxiliary losses (excludes L_uv).
        breakdown : dict[str, tf.Tensor]
            Raw (un-weighted) individual loss tensors.
        """
        mu_tf  = tf.constant(cfg.mu,  dtype=tf.float32, name='phys_mu')
        sig_tf = tf.constant(cfg.sig, dtype=tf.float32, name='phys_sig')

        speed_hr = _denorm_speed(x_HR, mu_tf, sig_tf)   # [batch, H, W]
        speed_sr = _denorm_speed(x_SR, mu_tf, sig_tf)   # [batch, H, W]

        # --- L1: Physical speed MSE ---
        # Penalises mean-squared error between physical wind speeds.
        L_speed = tf.reduce_mean((speed_hr - speed_sr) ** 2)

        # --- L2: Speed gradient-magnitude MSE ---
        # Penalises mismatch in the spatial sharpness of the speed field.
        gm_hr = _gradmag(speed_hr)
        gm_sr = _gradmag(speed_sr)
        L_grad = tf.reduce_mean((gm_hr - gm_sr) ** 2)

        # --- L3: Wind-power-density proxy MAE (speed^3) ---
        # speed^3 is proportional to the instantaneous kinetic power flux.
        # MAE used (not MSE) to keep magnitudes in a tractable range.
        # Disabled by default (lambda_wpd=0) because raw magnitudes are large.
        wpd_hr = speed_hr ** 3
        wpd_sr = speed_sr ** 3
        L_wpd = tf.reduce_mean(tf.abs(wpd_hr - wpd_sr))

        # --- L4: Soft level-set / topology-inspired superlevel-set loss ---
        # For each physical threshold tau, approximate the exceedance mask with
        # a sigmoid.  MSE between HR and SR soft masks penalises mismatch in
        # the spatial extent of high-speed regions without hard comparisons.
        # This is NOT a merge-tree loss; it is a topology-inspired proxy.
        k = cfg.levelset_temperature
        tau_losses = []
        for tau in cfg.levelset_thresholds:
            mask_hr = tf.sigmoid(k * (speed_hr - tau))
            mask_sr = tf.sigmoid(k * (speed_sr - tau))
            tau_losses.append(tf.reduce_mean((mask_hr - mask_sr) ** 2))
        L_levelset = tf.reduce_mean(tf.stack(tau_losses))

        # --- L5: Critical-value / topological-extrema proxy ---
        # Concentrates MSE on high-speed GT local maxima (superlevel-set births).
        # Disabled by default (lambda_crit=0).
        L_crit = _critical_value_loss(
            speed_hr, speed_sr,
            high_z         = cfg.crit_high_z,
            include_minima = cfg.crit_include_minima,
            low_z          = cfg.crit_low_z,
            pool           = cfg.crit_pool,
        )

        breakdown = {
            'L_speed':    L_speed,
            'L_grad':     L_grad,
            'L_wpd':      L_wpd,
            'L_levelset': L_levelset,
            'L_crit':     L_crit,
        }

        weighted_total = (
            cfg.lambda_speed    * L_speed    +
            cfg.lambda_grad     * L_grad     +
            cfg.lambda_wpd      * L_wpd      +
            cfg.lambda_levelset * L_levelset +
            cfg.lambda_crit     * L_crit
        )

        return weighted_total, breakdown


    def generator(self, x, r, is_training=False, reuse=False):
        if is_training:
            N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.get_shape()[3]

        k, stride = 3, 1
        output_shape = [N, h+2*k, w+2*k, -1]

        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                C_in, C_out = C, 64
                output_shape[-1] = C_out
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.nn.relu(x)

            skip_connection = x

            # B residual blocks
            C_in, C_out = C_out, 64
            output_shape[-1] = C_out
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.nn.relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('deconv2'):
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.add(x, skip_connection)

            # Super resolution scaling
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('deconv{}'.format(i+3)):
                    output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C_out]
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.relu(x)

                r_prod *= r_i

            output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C]
            with tf.variable_scope('deconv_out'):
                x = deconv_layer_2d(x, [k, k, C, C_in], output_shape, stride, k)

        return x


    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x

    def compute_losses(self, x_HR, x_SR, d_HR, d_SR, alpha_advers=0.001, isGAN=False):

        content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])

        if isGAN:
            g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))

            d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                                    labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

            advers_perf = [tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32)), # % true positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32)), # % true negative
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32)), # % false positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))] # % false negative

            g_loss = tf.reduce_mean(content_loss) + alpha_advers*tf.reduce_mean(g_advers_loss)
            d_loss = tf.reduce_mean(d_advers_loss)

            return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            return tf.reduce_mean(content_loss)
