## Starting TensorBoard

1. Launch TensorBoard by pointing it to your logs directory:
   ```bash
   tensorboard --logdir=./logs
   ```
   
   If your logs are in a different directory than the default `./logs`, specify that path instead:
   ```bash
   tensorboard --logdir=/full/path/to/logs
   ```

2. You should see output similar to:
   ```
   TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
   ```

3. Open a web browser and go to:
   ```
   http://localhost:6006
   ```

## Alternative TensorBoard Commands

If you're running this on a remote server (like a cloud VM):
```bash
tensorboard --logdir=./logs --host 0.0.0.0 --port 6006
```
Then access it from your local machine at `http://<server-ip>:6006`

If port 6006 is already in use:
```bash
tensorboard --logdir=./logs --port 6007
```

## Finding Your Log Files

Based on your TensorLogger implementation, your logs will be stored in a directory structure like:
```
./logs/YYYY-MM-DD_HH-MM-SS/
```

For example, with today's date (2025-04-01), it might be:
```
./logs/2025-04-01_10-54-19/
```

## What You'll See in TensorBoard

When you open TensorBoard in your browser, you'll see tabs for:

1. **Time Series** - Shows your metrics over time:
   - `train/g_loss` - Generator loss
   - `train/d_loss` - Discriminator loss
   - `train/g_learning_rate` - Generator learning rate
   - `train/d_learning_rate` - Discriminator learning rate
   - `train/g_d_ratio` - Ratio of generator to discriminator loss

2. **Distributions** - Shows distributions of your metrics over time

3. **Histograms** - Shows detailed metric distributions 

## Tips for Using TensorBoard

1. **Compare Runs**: If you have multiple runs, TensorBoard will show them as different colored lines, making it easy to compare experiments.

2. **Smoothing**: Use the "Smoothing" slider in the UI to reduce noise in your graphs.

3. **Custom Layout**: You can create custom layouts by selecting specific metrics and clicking "Save as View" in TensorBoard.

4. **Export Data**: You can export data as CSV by clicking on the download button in the graph view.