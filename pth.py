import timm

# Load the GhostNet model
backbone = timm.create_model('ghostnetv2_100', pretrained=True)

# Print the model architecture
print(backbone)

