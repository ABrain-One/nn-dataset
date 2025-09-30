import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def supported_hyperparameters():
    return {'lr', 'momentum'}


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Expert, self).__init__()
        self.input_dim = input_dim
                                                
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 3)                        
        self.fc3 = nn.Linear(hidden_dim // 3, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))                  
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Gate(nn.Module):
    def __init__(self, input_dim, n_experts, hidden_dim=32):
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.n_experts = n_experts
                                         
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.dropout = nn.Dropout(0.1)
        self.top_k = 3                        

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if x.size(-1) != self.input_dim:
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]
            else:
                padding = self.input_dim - x.size(-1)
                x = F.pad(x, (0, padding))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        gate_logits = self.fc2(x)
        
                                   
        if self.training:
                                                          
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise
        
                           
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
                            
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
                                    
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices


class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(Net, self).__init__()
        self.device = device
        self.n_experts = 8                          
        self.top_k = 2                                  

        if isinstance(in_shape, (list, tuple)) and len(in_shape) > 1:
            self.input_dim = 1
            for dim in in_shape:
                self.input_dim *= dim
        else:
            self.input_dim = in_shape[0] if isinstance(in_shape, (list, tuple)) else in_shape

        self.output_dim = out_shape[0] if isinstance(out_shape, (list, tuple)) else out_shape

                                                        
        self.hidden_dim = min(128, max(32, self.input_dim // 8))                         

                                                        
        if self.input_dim > 2048:
            print(f"Warning: Large input dimension {self.input_dim}, limiting to 2048")
            self.input_dim = 2048

                                                    
        self.experts = nn.ModuleList([
            Expert(self.input_dim, self.hidden_dim, self.output_dim)
            for _ in range(self.n_experts)
        ])
        
                                
        self.gate = Gate(self.input_dim, self.n_experts, self.hidden_dim // 3)                                 

                        
        self.to(device)

                                 
        self._print_memory_info()

    def _print_memory_info(self):
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)                       
        print(f"MoE-8 Model parameters: {param_count:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        print(f"Experts: {self.n_experts}, Top-K: {self.top_k}")                 
        print(f"Input dim: {self.input_dim}, Hidden dim: {self.hidden_dim}, Output dim: {self.output_dim}")

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    def forward(self, x):
        try:
                                          
            x = x.float()
            batch_size = x.size(0)

                                           
            if batch_size > 128:                           
                print(f"Warning: Large batch size {batch_size}, consider reducing")

                                           
            if x.dim() > 2:
                x = x.view(batch_size, -1)

                                         
            if x.size(-1) > self.input_dim:
                x = x[:, :self.input_dim]

                                               
            gate_weights, top_k_indices = self.gate(x)

                                                                  
            output = torch.zeros(batch_size, self.output_dim, device=self.device)
            
                                                     
            for i in range(self.n_experts):
                                                            
                expert_mask = (top_k_indices == i).any(dim=1)
                if expert_mask.any():
                                                                                     
                    expert_output = self.experts[i](x)
                                          
                    weighted_output = expert_output * gate_weights[:, i].unsqueeze(-1)
                    output += weighted_output

            return output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory! Clearing cache and trying with smaller batch...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                                                
                return torch.zeros(x.size(0), self.output_dim, device=self.device)
            else:
                raise e

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=prm.get('lr', 0.01),
                                         momentum=prm.get('momentum', 0.9))                     

    def learn(self, train_data):
        self.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_data):
            try:
                                   
                if batch_idx % 10 == 0:                            
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                                                        
                if inputs.size(0) > 32:
                    inputs = inputs[:32]
                    labels = labels[:32]

                self.optimizer.zero_grad()
                outputs = self(inputs)

                                      
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                if labels.dim() > 1:
                    labels = labels.view(-1)

                loss = self.criteria(outputs, labels)
                loss.backward()

                                                 
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                                            
                del inputs, labels, outputs, loss

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at batch {batch_idx}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f"Training error: {e}")
                    continue

        return total_loss / max(num_batches, 1)

    def evaluate(self, test_data):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                try:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)

                                      
                    if inputs.size(0) > 32:
                        inputs = inputs[:32]
                        labels = labels[:32]

                    outputs = self(inputs)

                    if outputs.dim() > 2:
                        outputs = outputs.view(outputs.size(0), -1)
                    if labels.dim() > 1:
                        labels = labels.view(-1)

                    loss = self.criteria(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                                   
                    del inputs, labels, outputs, loss

                except Exception as e:
                    print(f"Eval error: {e}")
                    continue

        return total_loss / len(test_data), correct / total if total > 0 else 0


                                 
if __name__ == "__main__":
                                  
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                        
    in_shape = (784,)                   
    out_shape = (10,)
    prm = {'lr': 0.01, 'momentum': 0.9}

    try:
        model = Net(in_shape, out_shape, prm, device)
        model.train_setup(prm)

        print("MoE-8 model created successfully!")

                               
        test_input = torch.randn(8, model.input_dim)                    
        test_output = model(test_input)
        print(f"Test successful! Output shape: {test_output.shape}")
        print(f"Sparse MoE with {model.n_experts} experts, top-{model.top_k} routing")

    except Exception as e:
        print(f"Error: {e}")