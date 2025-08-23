/*
 * Simple Image Canvas (ComfyUI)
 * Display two images using Fabric.js: base image + reference image (scaled to 100px max)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fabric } from "./fabric.js";

app.registerExtension({
  name: "omini.kontext_editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "OminiKontextEditor") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

                  // Initialize saved reference image state
      if (!this.savedRefImageState) {
        this.savedRefImageState = { left: null, top: null, scaleX: null, scaleY: null, angle: null };
      }

      // Utility function to send settings to backend
      const sendSettings = async (settings) => {
        try {
          const response = await api.fetchApi("/omini_kontext_editor/update_subject_settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ unique_id: this.id, settings })
          });
        } catch (error) {
          // Silent error handling
        }
      };

      // Send null settings to clear existing settings
      setTimeout(() => {
        sendSettings(null);
      }, 1000);

      // Create UI container
      const container = document.createElement("div");
      container.style.cssText = "display: flex; flex-direction: column; gap: 6px; width: 100%;";

      // Create stage for canvas
      const stage = document.createElement("div");
      stage.style.cssText = "position: relative; width: 100%; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;";

      const overlay = document.createElement("canvas");
      overlay.style.cssText = "position: relative; display: block;";

      stage.append(overlay);
      container.append(stage);

      // Add DOM widget
      this.addDOMWidget("omini_kontext_editor", "omini_kontext_editor", container, {
        getValue() {},
        setValue() {},
      });

      let fabricCanvas = null;

      // Function to send subject settings
      const sendSubjectSettings = () => {
        if (!fabricCanvas) return;
        
        const refImage = fabricCanvas.getObjects().find(obj => obj.selectable && obj.evented);
        if (!refImage) return;
        
        const settings = {
          left: refImage.left,
          top: refImage.top,
          scaleX: refImage.scaleX,
          scaleY: refImage.scaleY,
          angle: refImage.angle,
          canvasWidth: fabricCanvas.width,
          canvasHeight: fabricCanvas.height,
          overallScale: this.overallScale
        };
        
        sendSettings(settings);
      };

      // Function to calculate canvas dimensions
      const calculateCanvasDimensions = (imgWidth, imgHeight, maxSize = 500) => {
        const aspect = imgWidth / imgHeight;
        let canvasWidth, canvasHeight;
        
        if (imgWidth >= imgHeight) {
          canvasWidth = Math.min(imgWidth, maxSize);
          canvasHeight = Math.round(canvasWidth / aspect);
          if (canvasHeight > maxSize) {
            canvasHeight = maxSize;
            canvasWidth = Math.round(canvasHeight * aspect);
          }
        } else {
          canvasHeight = Math.min(imgHeight, maxSize);
          canvasWidth = Math.round(canvasHeight * aspect);
          if (canvasWidth > maxSize) {
            canvasWidth = maxSize;
            canvasHeight = Math.round(canvasWidth * aspect);
          }
        }
        
        return { canvasWidth, canvasHeight };
      };

      // Function to load and setup images
      const setupImages = async (background, subject, subject_settings) => {
        // Clean up existing canvas
        if (fabricCanvas) {
          fabricCanvas.dispose();
        }
        
        // Initialize Fabric.js canvas
        fabricCanvas = new fabric.Canvas(overlay);
        fabricCanvas.selection = true;
        fabricCanvas.preserveObjectStacking = true;
        fabricCanvas.isDrawingMode = false;
        fabricCanvas.stopContextMenu = true;
        fabricCanvas.defaultCursor = 'default';
        fabricCanvas.selectionColor = 'rgba(0,123,255,0.3)';
        fabricCanvas.selectionBorderColor = 'rgba(0,123,255,1)';
        
        try {
          // Load background image
          const baseImg = await new Promise((resolve) => {
            fabric.Image.fromURL(background, resolve, { crossOrigin: 'anonymous' });
          });
          
          const { canvasWidth, canvasHeight } = calculateCanvasDimensions(
            baseImg.width || 512, 
            baseImg.height || 512
          );
          
          fabricCanvas.setDimensions({ width: canvasWidth, height: canvasHeight });
          this.overallScale = canvasWidth / baseImg.width;
          
          // Add background image (non-selectable)
          baseImg.set({
            selectable: false,
            evented: false,
            scaleX: this.overallScale,
            scaleY: this.overallScale
          });
          baseImg.setCoords();
          fabricCanvas.add(baseImg);
          
          // Update stage dimensions
          stage.style.width = `${canvasWidth}px`;
          stage.style.height = `${canvasHeight}px`;
          overlay.style.width = `${canvasWidth}px`;
          overlay.style.height = `${canvasHeight}px`;
          
          // Load subject image if available
          if (subject) {
            const refImg = await new Promise((resolve) => {
              fabric.Image.fromURL(subject, resolve, { crossOrigin: 'anonymous' });
            });
            
            // Calculate scale to fit within canvas bounds
            const maxSide = Math.min(canvasWidth/2, canvasHeight/2);
            const scaleX = maxSide / refImg.width;
            const scaleY = maxSide / refImg.height;
            const scale = Math.min(scaleX, scaleY);
            
            refImg.scale(scale);
            
            // Determine initial position and scale
            let settings = { top: null, left: null, scaleX: null, scaleY: null, angle: null };
            
            if (subject_settings) {
              settings = {
                top: subject_settings.top,
                left: subject_settings.left,
                scaleX: subject_settings.scaleX,
                scaleY: subject_settings.scaleY,
                angle: subject_settings.angle || 0
              };
            } else if (this.savedRefImageState.top !== null) {
              settings = { ...this.savedRefImageState };
            } else {
              // Default center position
              settings = {
                top: (canvasHeight - refImg.getScaledHeight()) / 2,
                left: (canvasWidth - refImg.getScaledWidth()) / 2,
                scaleX: scale,
                scaleY: scale,
                angle: 0
              };
            }

            // Save state
            this.savedRefImageState = { ...settings };
            
            // Configure subject image
            refImg.set({
              left: settings.left,
              top: settings.top,
              scaleX: settings.scaleX,
              scaleY: settings.scaleY,
              angle: settings.angle,
              selectable: true,
              evented: true,
              hasControls: true,
              hasBorders: true,
              hasRotatingPoint: true,
              lockRotation: false
            });
            
            fabricCanvas.add(refImg);
            refImg.bringToFront();
            refImg.setCoords();
            
            // Add event listeners
            const updateSettings = () => {
              this.savedRefImageState.left = refImg.left;
              this.savedRefImageState.top = refImg.top;
              this.savedRefImageState.scaleX = refImg.scaleX;
              this.savedRefImageState.scaleY = refImg.scaleY;
              this.savedRefImageState.angle = refImg.angle;
              console.log("savedRefImageState", this.savedRefImageState);
              sendSubjectSettings();
            };
            
            refImg.on('moving', updateSettings);
            refImg.on('scaling', updateSettings);
            refImg.on('rotating', updateSettings);
            
            fabricCanvas.requestRenderAll();
          }
          
          fabricCanvas.renderAll();
          
          // Update node size
          const newSize = [canvasWidth + 20, canvasHeight + 120];
          if (!this.size || this.size[0] !== newSize[0] || this.size[1] !== newSize[1]) {
            this.setSize(newSize);
          }
          
        } catch (error) {
          // Silent error handling
        }
      };

      // Listen for background images from server
      api.addEventListener("simpledraw_bg", async ({ detail }) => {
        const { unique_id, background, subject, subject_settings } = detail || {};
        if (String(unique_id) !== String(this.id)) return;
        
        await setupImages(background, subject, subject_settings);
        sendSubjectSettings();
      });

      return r;
    };
  },
});
