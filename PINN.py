import torch
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load in the data (currently hardcoded to our DNS dataset)
def load_data():
    # Load grid points (y-coordinates)
    y_DNS = np.loadtxt('y.txt')  # Shape: (512,)
    y_min, y_max = -1.0, 1.0  # DNS y-range is [-1,1]
    y_normalized = (y_DNS - y_min) / (y_max - y_min)  # Map to [0,1]

    # Load mean profiles
    profile_data = np.loadtxt('profiles.txt', skiprows=2, max_rows=257)  # Shape: (257, columns)
    u_DNS = profile_data[:, 1]  # Velocity
    p_DNS = profile_data[:, 4]  # Pressure
    
    # Load friction velocity (u_tau) history
    u_tau_data = np.loadtxt('re-tau.txt', skiprows=2)  # Shape: (Nt, 2)
    u_tau = u_tau_data[:, 1]  # Time-varying u_tau

    # Calculate scaling factors
    u_scale = np.max(np.abs(u_DNS))  
    p_scale = np.max(np.abs(p_DNS)) if np.max(np.abs(p_DNS)) != 0 else 1.0
    
    # Return normalized data
    return (
        y_normalized[:257],  # Match profile data length
        u_DNS/u_scale, 
        p_DNS/p_scale, 
        u_tau/1000.0,  # Scale u_tau if needed
        y_min, 
        y_max
    )


# Class for our PINN
class PINN(nn.Module):
    def __init__(self, num_layers=10, hidden_size=32):
        super(PINN, self).__init__()
        
        # More complex network architecture with residual connections
        self.base_layers = nn.ModuleList()
        
        # Input layer
        self.base_layers.append(nn.Linear(3, hidden_size))
        self.base_layers.append(nn.GELU())
        
        # Residual blocks
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.base_layers.append(block)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 3)
        
    def forward(self, x):
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Residual connection through the network
        h = self.base_layers[0](x)
        for layer in self.base_layers[1:-1]:
            if isinstance(layer, nn.Sequential):
                h = h + layer(h)
            else:
                h = layer(h)
        
        return self.output_layer(h)


# Class for our 
class PhysicsLoss:
    def __init__(self, nu=0.01, rho=1.0, 
                 u_scale=14.2, p_scale=1.0,  # DNS scaling factors
                 penalize_residuals=True,
                 enforce_incompressibility=True,
                 boundary_penalty=True):
        
        # Physical parameters
        self.nu = nu                  # Kinematic viscosity
        self.rho = rho                # Fluid density
        self.u_scale = u_scale        # DNS velocity scaling factor
        self.p_scale = p_scale        # DNS pressure scaling factor
        
        # Loss configuration
        self.penalize_residuals = penalize_residuals
        self.enforce_incompressibility = enforce_incompressibility
        self.boundary_penalty = boundary_penalty
    
    def compute_loss(self, model, x, y, t):
        # Enable gradient tracking
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        # Network predictions (scaled to DNS units)
        inputs = torch.cat([x, y, t], dim=1)
        outputs = model(inputs)
        vx = outputs[:, 0] * self.u_scale  # Scale velocity
        vy = outputs[:, 1] * self.u_scale
        p = outputs[:, 2] * self.p_scale   # Scale pressure
        
        
        # Gradient computation function
        def gradient(output, inputs):
            return torch.autograd.grad(
                output, inputs, 
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True
            )[0]
        
        # First derivatives
        dvx_dx = gradient(vx, x)
        dvx_dy = gradient(vx, y)
        dvx_dt = gradient(vx, t)
        
        dvy_dx = gradient(vy, x)
        dvy_dy = gradient(vy, y)
        dvy_dt = gradient(vy, t)
        
        dp_dx = gradient(p, x)
        dp_dy = gradient(p, y)
        
        # Second derivatives
        dvx_dxx = gradient(dvx_dx, x)
        dvx_dyy = gradient(dvx_dy, y)
        dvy_dxx = gradient(dvy_dx, x)
        dvy_dyy = gradient(dvy_dy, y)
        
       # Physics-based losses
        losses = {}
        
        # Continuity equation
        if self.enforce_incompressibility:
            continuity = dvx_dx + dvy_dy
            losses['continuity'] = torch.mean(continuity**2)
        
        # Momentum equations
        if self.penalize_residuals:
            ns_x = (dvx_dt + vx*dvx_dx + vy*dvx_dy + 
                   (1/self.rho)*dp_dx - self.nu*(dvx_dxx + dvx_dyy))
            ns_y = (dvy_dt + vx*dvy_dx + vy*dvy_dy + 
                   (1/self.rho)*dp_dy - self.nu*(dvy_dxx + dvy_dyy))
            
            losses['momentum_x'] = torch.mean(ns_x**2)
            losses['momentum_y'] = torch.mean(ns_y**2)
        
        # Pressure gradients
        losses['pressure'] = torch.mean(dp_dx**2 + dp_dy**2)
        
        # Boundary conditions - FIXED VERSION
        if self.boundary_penalty:
            # Create boundary masks (using .squeeze() to handle 1D case)
            x_boundary = ((x.squeeze() < 0.01) | (x.squeeze() > 0.99))
            y_boundary = ((y.squeeze() < 0.01) | (y.squeeze() > 0.99))
            
            # Only calculate loss if there are boundary points
            if x_boundary.any():
                losses['boundary_x'] = torch.mean(vx[x_boundary]**2)
            else:
                losses['boundary_x'] = torch.tensor(0.0, device=device)
                
            if y_boundary.any():
                losses['boundary_y'] = torch.mean(vy[y_boundary]**2)
            else:
                losses['boundary_y'] = torch.tensor(0.0, device=device)
        
        # Weighted loss components
        total_loss = (
            losses.get('continuity', 0) * 10.0 +
            losses.get('momentum_x', 0) * 1.0 +
            losses.get('momentum_y', 0) * 1.0 +
            losses.get('pressure', 0) * 0.5 +
            losses.get('boundary_x', 0) * 1.0 +
            losses.get('boundary_y', 0) * 1.0
        )
        
        return total_loss


def train_model(model, loss_fn, optimizer, 
                x_min=0.0, x_max=1.0,
                y_min=-1.0, y_max=1.0,  # DNS y-range is [-1,1]
                t_min=0.0, t_max=1.0,
                batch_size=2000, epochs=500):
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample points in DNS-scaled domain
        x = torch.rand(batch_size, 1, device=device) * (x_max - x_min) + x_min
        y = torch.rand(batch_size, 1, device=device) * (y_max - y_min) + y_min
        t = torch.rand(batch_size, 1, device=device) * (t_max - t_min) + t_min
        
        # Compute and backpropagate loss
        loss = loss_fn.compute_loss(model, x, y, t)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model

def plot_results(model, t=0.0, n_points=100):
    # Create more detailed grid
    x = torch.linspace(0, 1, n_points, device=device)
    y = torch.linspace(0, 1, n_points, device=device)
    
    # Create meshgrid with indexing='xy'
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten and prepare inputs
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = torch.full_like(x_flat, t)
    
    # Combine inputs correctly
    inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)
    
    # Reshape outputs and convert to numpy
    vx = outputs[:, 0].reshape(n_points, n_points).cpu().numpy()
    vy = outputs[:, 1].reshape(n_points, n_points).cpu().numpy()
    p = outputs[:, 2].reshape(n_points, n_points).cpu().numpy()
    
    # Convert meshgrid to numpy for plotting
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    plt.figure(figsize=(18, 6))
    
    # Velocity magnitude with improved quiver plot
    plt.subplot(1, 3, 1)
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    plt.imshow(velocity_magnitude, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
    plt.colorbar(label='Velocity Magnitude')
    
    # Add quiver plot for vector field
    skip = (slice(None, None, 5), slice(None, None, 5))
    plt.quiver(X_np[skip], Y_np[skip], 
               vx[skip], vy[skip], 
               color='white', alpha=0.7)
    
    plt.title(f'Velocity Field at t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Pressure field with improved colormap
    plt.subplot(1, 3, 2)
    plt.imshow(p, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
    plt.colorbar(label='Pressure')
    plt.title(f'Pressure Field at t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Vorticity with higher resolution
    plt.subplot(1, 3, 3)
    dx = x[1].cpu() - x[0].cpu()
    dy = y[1].cpu() - y[0].cpu()
    dvx_dy, dvx_dx = np.gradient(vx, dy, dx)
    dvy_dy, dvy_dx = np.gradient(vy, dy, dx)
    vorticity = dvy_dx - dvx_dy
    plt.imshow(vorticity, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title(f'Vorticity at t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()

    # Save the plot
    plt.savefig('PINN_plot.png')

    plt.show()



def animation(model, t_min=0.0, t_max=1.0, n_frames=20, n_points=100, filename="pinn_evolution.gif"):
    """
    Create an animation of the PINN results over time.
    
    Parameters:
    - model: Trained PINN model
    - t_min: Start time
    - t_max: End time
    - n_frames: Number of frames in the animation
    - n_points: Resolution of the grid
    - filename: Output GIF filename
    """
    from PIL import Image
    import os
    
    # Create temporary directory for frames
    os.makedirs("temp_frames", exist_ok=True)
    
    # Generate time points
    time_points = torch.linspace(t_min, t_max, n_frames, device=device)
    
    # Create grid
    x = torch.linspace(0, 1, n_points, device=device)
    y = torch.linspace(0, 1, n_points, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    frame_files = []
    
    for i, t in enumerate(time_points):
        #print(f"Generating frame {i+1}/{n_frames} at t={t.item():.2f}")
        
        # Prepare inputs with current time
        t_flat = torch.full_like(x_flat, t.item())
        inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(inputs)
        
        # Reshape outputs
        vx = outputs[:, 0].reshape(n_points, n_points).cpu().numpy()
        vy = outputs[:, 1].reshape(n_points, n_points).cpu().numpy()
        p = outputs[:, 2].reshape(n_points, n_points).cpu().numpy()
        
        # Convert meshgrid to numpy
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        
        # Create figure
        fig = plt.figure(figsize=(18, 6))
        
        # Velocity magnitude
        plt.subplot(1, 3, 1)
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        plt.imshow(velocity_magnitude, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        plt.colorbar(label='Velocity Magnitude')
        skip = (slice(None, None, 5), slice(None, None, 5))
        plt.quiver(X_np[skip], Y_np[skip], vx[skip], vy[skip], color='white', alpha=0.7)
        plt.title(f'Velocity Field at t={t.item():.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Pressure field
        plt.subplot(1, 3, 2)
        plt.imshow(p, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm')
        plt.colorbar(label='Pressure')
        plt.title(f'Pressure Field at t={t.item():.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Vorticity
        plt.subplot(1, 3, 3)
        dx = x[1].cpu() - x[0].cpu()
        dy = y[1].cpu() - y[0].cpu()
        dvx_dy, dvx_dx = np.gradient(vx, dy, dx)
        dvy_dy, dvy_dx = np.gradient(vy, dy, dx)
        vorticity = dvy_dx - dvx_dy
        plt.imshow(vorticity, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
        plt.colorbar(label='Vorticity')
        plt.title(f'Vorticity at t={t.item():.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = f"temp_frames/frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frame_files.append(frame_path)
        plt.close()
    
    # Create GIF from frames
    images = [Image.open(f) for f in frame_files]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=200,  # milliseconds per frame
        loop=0
    )
    
    # Clean up temporary files
    for f in frame_files:
        os.remove(f)
    os.rmdir("temp_frames")
    
    print(f"Animation saved as {filename}")



# main function 
if __name__ == "__main__":

    # GPU configuration
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Compute derivatives with autograd 
    def gradient(output, inputs):
        return torch.autograd.grad(
            output, inputs, 
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]


    # Load DNS data
    y_DNS, u_DNS, p_DNS, u_tau, y_min, y_max = load_data()
    nu = 0.01  # Kinematic viscosity 

    # Set random seed for reproducibility
    torch.manual_seed(45)
    np.random.seed(45)
    
    # Initialize improved model and optimizer
    model = PINN(num_layers=10, hidden_size=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # Enhanced physics loss with DNS data
    physics_loss = PhysicsLoss(
        nu=nu,
        rho=1.0,
        penalize_residuals=True,
        enforce_incompressibility=True,
        boundary_penalty=True
    )
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    model = train_model(model, physics_loss, optimizer, 
                              x_min=0.0, x_max=1.0,
                              y_min=y_min, y_max=y_max,
                              t_min=0.0, t_max=1.0,
                              batch_size=1024, epochs=1000)
    total_time = time.time() - start_time
    print(f"Training Time: {total_time}")
    
    # Save model
    #torch.save(model.state_dict(), "improved_navier_stokes_pinn.pth")
    
    def create_eval_tensors():
        return (
            torch.rand(2000, 1, device=device, requires_grad=True),
            torch.rand(2000, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min,
            torch.rand(2000, 1, device=device, requires_grad=True)
        )
    
    if False:  # Set to False to skip tuning
        print("\n=== HYPERPARAMETER TUNING ===")
        print("\n=== This will affect the out put of the final model eval so disregard it===")
        
        # Test configurations
        hp_configs = {
            'batch_size': [512, 1024, 2048],
            'epochs': [32, 64, 128, 256, 512], 
            'lr': [1e-4, 5e-4, 1e-3, 5e-3],
            'num_layers': [6, 8, 10, 12],
            'hidden_size': [32, 64, 128]
        }
        
        # Reduced test epochs for tuning
        tune_epochs = 40  
        best_loss = float('inf')
        best_params = {}
        
        # Grid search implementation
        from itertools import product
        for bs, lr, nl, hs in product(
            hp_configs['batch_size'],
            hp_configs['lr'],
            hp_configs['num_layers'],
            hp_configs['hidden_size']
        ):
            print(f"\nTesting: bs={bs}, lr={lr}, layers={nl}, hidden={hs}")
            
            # Reinitialize with current params
            model = PINN(num_layers=nl, hidden_size=hs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            
            # Short training run
            start_time = time.time()
            model = train_model(model, physics_loss, optimizer,
                              x_min=0.0, x_max=1.0,
                              y_min=y_min, y_max=y_max,
                              t_min=0.0, t_max=1.0,
                              batch_size=bs, epochs=tune_epochs)
            
            # Create fresh evaluation tensors for each configuration
            x_eval, y_eval, t_eval = create_eval_tensors()
            
            # Evaluation
            with torch.no_grad():
                model.eval()
                with torch.enable_grad():  # Enable grad for physics loss computation
                    eval_loss = physics_loss.compute_loss(model, x_eval, y_eval, t_eval)
                print(f"Test Loss: {eval_loss.item():.6f} | Time: {time.time()-start_time:.2f}s")
                
                # Track best
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    best_params = {
                        'batch_size': bs,
                        'lr': lr,
                        'num_layers': nl,
                        'hidden_size': hs
                    }
        
        # Results summary
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"Best Loss: {best_loss:.6f}")
        for k, v in best_params.items():
            print(f"{k:>12}: {v}")
        print("Note: Epochs should be set as high as possible (500+) for final training")

    # Create fresh evaluation tensors for final evaluation
    x_eval, y_eval, t_eval = create_eval_tensors()

    # Model performance
    print("\nMODEL PERFORMANCE EVALUATION\n")
    
    # Final Loss Evaluation 
    x_eval = torch.rand(2000, 1, device=device, requires_grad=True)
    y_eval = torch.rand(2000, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min
    t_eval = torch.rand(2000, 1, device=device, requires_grad=True)
    
    # Compute loss without affecting model parameters
    with torch.no_grad():  # Don't track gradients for model parameters
        model.eval()
        # Need to enable grad for the inputs to compute physics loss
        with torch.enable_grad():  # Temporarily enable grad for input tensors
            eval_loss = physics_loss.compute_loss(model, x_eval, y_eval, t_eval)
        print(f"\nFinal Physics Loss: {eval_loss.item():.6f}")
    
# Plot PINN predictions
try:
    # Select points along y-axis at x=0.5, t=0
    y_DNS_tensor = torch.tensor(y_DNS, dtype=torch.float32, device=device).unsqueeze(1)
    x_DNS_tensor = torch.full_like(y_DNS_tensor, 0.5)  # Middle of domain in x
    t_DNS_tensor = torch.zeros_like(y_DNS_tensor)      # Initial time
    
    # Get model predictions
    inputs = torch.cat([x_DNS_tensor, y_DNS_tensor, t_DNS_tensor], dim=1)
    with torch.no_grad():
        outputs = model(inputs)
    
    # Get predictions
    vx_pred = outputs[:, 0].cpu().numpy()
    vy_pred = outputs[:, 1].cpu().numpy()
    p_pred = outputs[:, 2].cpu().numpy()
    
    # Calculate errors 
    u_error = np.mean(np.abs(vx_pred - u_DNS))
    p_error = np.mean(np.abs(p_pred - p_DNS))
    
    # Print the stored MAE values 
    print(f"\nModel Performance Metrics:")
    print(f"  Velocity MAE: {u_error} ")
    print(f"  Pressure MAE: {p_error}")
    
    # Plot velocity and pressure profiles
    plt.figure(figsize=(12, 5))
    
    # Velocity plot
    plt.subplot(1, 2, 1)
    plt.plot(y_DNS, vx_pred, 'b-', label='PINN Prediction')
    plt.xlabel('Normalized y-coordinate')
    plt.ylabel('Normalized u velocity')
    plt.legend()
    plt.title('PINN Velocity Profile (x=0.5, t=0)')
    
    # Pressure plot
    plt.subplot(1, 2, 2)
    plt.plot(y_DNS, p_pred, 'g-', label='PINN Prediction')
    plt.xlabel('Normalized y-coordinate')
    plt.ylabel('Normalized pressure')
    plt.legend()
    plt.title('PINN Pressure Profile (x=0.5, t=0)')
    
    plt.tight_layout()
    plt.savefig('PINN_predictions_only.png')
    plt.show()

except Exception as e:
    print(f"\nError plotting PINN predictions: {str(e)}")
    
# Physics Residuals Analysis
print("\nPhysics Residuals Analysis:")
# Need fresh tensors with requires_grad=True
x_res = torch.rand(1000, 1, device=device, requires_grad=True)
y_res = torch.rand(1000, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min
t_res = torch.rand(1000, 1, device=device, requires_grad=True)

# Compute without affecting model parameters
with torch.no_grad():
    model.eval()
    with torch.enable_grad():  # Enable grad for residual calculations
        inputs = torch.cat([x_res, y_res, t_res], dim=1)
        outputs = model(inputs)
        vx = outputs[:, 0:1]
        vy = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        # Compute derivatives
        dvx_dx = gradient(vx, x_res)
        dvx_dy = gradient(vx, y_res)
        dvy_dx = gradient(vy, x_res)
        dvy_dy = gradient(vy, y_res)
        
        # Continuity residual
        continuity_res = (dvx_dx + dvy_dy).abs().mean()
        print(f"  Continuity equation residual: {continuity_res.item():.6f}")
        
        # Momentum residuals
        dvx_dt = gradient(vx, t_res)
        dvy_dt = gradient(vy, t_res)
        dp_dx = gradient(p, x_res)
        dp_dy = gradient(p, y_res)
        
        # Second derivatives for viscous terms
        dvx_dxx = gradient(dvx_dx, x_res)
        dvx_dyy = gradient(dvx_dy, y_res)
        dvy_dxx = gradient(dvy_dx, x_res)
        dvy_dyy = gradient(dvy_dy, y_res)
        
        mom_x_res = (dvx_dt + vx*dvx_dx + vy*dvx_dy + dp_dx - nu*(dvx_dxx + dvx_dyy)).abs().mean()
        mom_y_res = (dvy_dt + vx*dvy_dx + vy*dvy_dy + dp_dy - nu*(dvy_dxx + dvy_dyy)).abs().mean()
        
        print(f"  X-momentum residual: {mom_x_res.item():.6f}")
        print(f"  Y-momentum residual: {mom_y_res.item():.6f}")


# plot results
plot_results(model, t=0.0)

# show animation
animation(model, t_min=0.0, t_max=1.0, n_frames=30, filename="navier_stokes_evolution.gif")
print("Animation complete!")
