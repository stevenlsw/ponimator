window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: false,
		autoplaySpeed: 3000,
    }

		// Temporarily show all dataset carousels for initialization
    $('.dataset-carousel').addClass('active-carousel');
    $('.single-pose-carousel').addClass('active-carousel');
    
		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
    
    // Hide all except Inter-X after initialization for each section
    $('.dataset-carousel').removeClass('active-carousel');
    $('#interx-carousel').addClass('active-carousel');
    
    $('.single-pose-carousel').removeClass('active-carousel');
    $('#single-pose-interx-carousel').addClass('active-carousel');

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

// Dataset switcher function for Interactive Pose Animation
function ChangeDataset(dataset) {
  console.log('Changing to dataset:', dataset);
  
  // Hide all carousels by removing active-carousel class
  $('.dataset-carousel').removeClass('active-carousel');
  
  // Show selected carousel by adding active-carousel class
  $('#' + dataset + '-carousel').addClass('active-carousel');
  
  console.log('Active carousel:', $('#' + dataset + '-carousel').length);
  
  // Update active button
  $('#dataset-selector li').removeClass('active');
  
  // Set active button - find the button that was clicked
  $('#dataset-selector li a').each(function() {
    if ($(this).attr('onclick').includes("'" + dataset + "'")) {
      $(this).parent('li').addClass('active');
    }
  });
}

// Dataset switcher function for Single-Person Pose Interaction Synthesis
function ChangeSinglePoseDataset(dataset) {
  console.log('Changing to single-pose dataset:', dataset);
  
  // Hide all single-pose carousels
  $('.single-pose-carousel').removeClass('active-carousel');
  
  // Show selected carousel
  $('#single-pose-' + dataset + '-carousel').addClass('active-carousel');
  
  console.log('Active single-pose carousel:', $('#single-pose-' + dataset + '-carousel').length);
  
  // Update active button
  $('#single-pose-dataset-selector li').removeClass('active');
  
  // Set active button
  $('#single-pose-dataset-selector li a').each(function() {
    if ($(this).attr('onclick').includes("'" + dataset + "'")) {
      $(this).parent('li').addClass('active');
    }
  });
}

// Comparison switcher function
function ChangeComparison(comparison) {
  console.log('Changing to comparison:', comparison);
  
  // Hide all comparison contents
  $('.comparison-content').removeClass('active-comparison');
  
  // Show selected comparison
  $('#' + comparison + '-comparison').addClass('active-comparison');
  
  console.log('Active comparison:', $('#' + comparison + '-comparison').length);
  
  // Update active button
  $('#comparison-selector li').removeClass('active');
  
  // Set active button
  $('#comparison-selector li a').each(function() {
    if ($(this).attr('onclick').includes("'" + comparison + "'")) {
      $(this).parent('li').addClass('active');
    }
  });
}
